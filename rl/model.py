"""History-aware action-id policy/value model for 4A4.

The model keeps the backend-facing class name `CardPolicyNetwork`, but its
candidate-action input is now a tensor of global action ids. It still returns
one score per currently legal action, so no global action mask is needed.
"""
try:
    import math
    import torch
    import torch.nn as nn
except ImportError:  # Allow rule AI to run on machines without torch.
    math = None
    torch = None
    nn = None

from rl.actions import ACTION_PAD_ID, ACTION_VOCAB_SIZE

RANK_COUNT = 15
HAND_CHANNELS = 5
HAND_MATRIX_SIZE = HAND_CHANNELS * RANK_COUNT
STATE_CONTEXT_DIM = 194
STATE_DIM = HAND_MATRIX_SIZE + STATE_CONTEXT_DIM
PERFECT_HAND_CHANNELS = HAND_CHANNELS * 4
PERFECT_HAND_MATRIX_SIZE = PERFECT_HAND_CHANNELS * RANK_COUNT
PERFECT_CONTEXT_DIM = 96
PERFECT_STATE_DIM = PERFECT_HAND_MATRIX_SIZE + PERFECT_CONTEXT_DIM
HISTORY_DIM = 5
HIDDEN_DIM = 512
HISTORY_LAYERS = 4


if nn is not None:
    def history_padding_mask(history_rows):
        if history_rows.numel() == 0:
            return torch.ones(
                history_rows.shape[:-1],
                dtype=torch.bool,
                device=history_rows.device)
        rel_empty = history_rows[..., :4].abs().sum(dim=-1) <= 1e-8
        ids = history_rows[..., 4].long()
        return rel_empty & (ids == ACTION_PAD_ID)


    def _apply_rope(x):
        """Apply rotary position encoding to q/k tensors.

        Shape: [batch, heads, seq_len, head_dim].
        """
        head_dim = x.shape[-1]
        rotary_dim = head_dim - (head_dim % 2)
        if rotary_dim <= 0:
            return x
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        half = rotary_dim // 2
        device = x.device
        dtype = x.dtype
        pos = torch.arange(x.shape[-2], device=device, dtype=dtype)
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, half, device=device, dtype=dtype) / half)
        )
        freqs = torch.einsum('l,d->ld', pos, inv_freq)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        even = x_rot[..., 0::2]
        odd = x_rot[..., 1::2]
        rotated = torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        rotated = rotated.flatten(-2)
        if x_pass.numel() == 0:
            return rotated
        return torch.cat([rotated, x_pass], dim=-1)


    class RotarySelfAttention(nn.Module):
        def __init__(self, hidden_dim=HIDDEN_DIM, nhead=8, dropout=0.1):
            super().__init__()
            if hidden_dim % nhead != 0:
                raise ValueError('hidden_dim must be divisible by nhead')
            self.hidden_dim = hidden_dim
            self.nhead = nhead
            self.head_dim = hidden_dim // nhead
            self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
            self.out = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, padding_mask=None):
            batch, seq_len, _ = x.shape
            qkv = self.qkv(x).view(
                batch, seq_len, 3, self.nhead, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = _apply_rope(q)
            k = _apply_rope(k)
            attn = torch.matmul(q, k.transpose(-2, -1))
            attn = attn / math.sqrt(self.head_dim)
            if padding_mask is not None:
                safe_mask = padding_mask.clone()
                all_pad = safe_mask.all(dim=1)
                if all_pad.any():
                    safe_mask[all_pad] = False
                attn = attn.masked_fill(
                    safe_mask[:, None, None, :], torch.finfo(attn.dtype).min)
            prob = torch.softmax(attn, dim=-1)
            prob = self.dropout(prob)
            out = torch.matmul(prob, v)
            out = out.transpose(1, 2).contiguous().view(
                batch, seq_len, self.hidden_dim)
            return self.out(out)


    class RotaryTransformerBlock(nn.Module):
        def __init__(self, hidden_dim=HIDDEN_DIM, nhead=8, dropout=0.1):
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.attn = RotarySelfAttention(hidden_dim, nhead, dropout)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.ff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, padding_mask=None):
            x = x + self.dropout(self.attn(self.norm1(x), padding_mask))
            x = x + self.dropout(self.ff(self.norm2(x)))
            return x


    class StateEncoder(nn.Module):
        def __init__(self, hidden_dim=HIDDEN_DIM):
            super().__init__()
            self.hand_conv = nn.Sequential(
                nn.Conv1d(HAND_CHANNELS, 64, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Flatten(),
                nn.Linear(64 * RANK_COUNT, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
            )
            self.context_mlp = nn.Sequential(
                nn.Linear(STATE_CONTEXT_DIM, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.GELU(),
            )
            self.out = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )

        def forward(self, state_vec):
            hand = state_vec[:, :HAND_MATRIX_SIZE].view(
                -1, HAND_CHANNELS, RANK_COUNT)
            context = state_vec[:, HAND_MATRIX_SIZE:]
            return self.out(torch.cat([
                self.hand_conv(hand),
                self.context_mlp(context),
            ], dim=-1))


    class PerfectStateEncoder(nn.Module):
        def __init__(self, hidden_dim=HIDDEN_DIM):
            super().__init__()
            self.hand_conv = nn.Sequential(
                nn.Conv1d(PERFECT_HAND_CHANNELS, 96, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(96, 96, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Flatten(),
                nn.Linear(96 * RANK_COUNT, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
            )
            self.context_mlp = nn.Sequential(
                nn.Linear(PERFECT_CONTEXT_DIM, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.GELU(),
            )
            self.out = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )

        def forward(self, perfect_state_vec):
            hands = perfect_state_vec[:, :PERFECT_HAND_MATRIX_SIZE].view(
                -1, PERFECT_HAND_CHANNELS, RANK_COUNT)
            context = perfect_state_vec[:, PERFECT_HAND_MATRIX_SIZE:]
            return self.out(torch.cat([
                self.hand_conv(hands),
                self.context_mlp(context),
            ], dim=-1))


    class RotaryHistoryEncoder(nn.Module):
        def __init__(self, action_embed, hidden_dim=HIDDEN_DIM,
                     nhead=8, num_layers=2, dropout=0.1):
            super().__init__()
            self.action_embed = action_embed
            self.input_proj = nn.Sequential(
                nn.Linear(hidden_dim + 4, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.layers = nn.ModuleList([
                RotaryTransformerBlock(hidden_dim, nhead, dropout)
                for _ in range(num_layers)
            ])
            self.out_norm = nn.LayerNorm(hidden_dim)

        def forward(self, history_rows):
            if history_rows.dim() == 2:
                history_rows = history_rows.unsqueeze(0)
            if history_rows.shape[1] == 0:
                pad = torch.zeros(
                    history_rows.shape[0], 1, HISTORY_DIM,
                    dtype=history_rows.dtype,
                    device=history_rows.device)
                pad[..., 4] = ACTION_PAD_ID
                history_rows = pad

            padding_mask = history_padding_mask(history_rows)
            rel = history_rows[..., :4].to(dtype=torch.float32)
            action_ids = history_rows[..., 4].long().clamp(
                min=0, max=ACTION_PAD_ID)
            action_emb = self.action_embed(action_ids)
            x = self.input_proj(torch.cat([rel, action_emb], dim=-1))
            for layer in self.layers:
                x = layer(x, padding_mask)
            x = self.out_norm(x)
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

            valid_counts = (~padding_mask).long().sum(dim=1)
            last_idx = (valid_counts - 1).clamp_min(0)
            batch_idx = torch.arange(x.shape[0], device=x.device)
            last = x[batch_idx, last_idx]
            empty = valid_counts == 0
            if empty.any():
                last = last.clone()
                last[empty] = 0.0
            return last


    class CardPolicyNetwork(nn.Module):
        """Shared symmetric RoPE Transformer model with action embeddings."""

        def __init__(self, state_dim=STATE_DIM,
                     action_dim=ACTION_VOCAB_SIZE,
                     history_dim=HISTORY_DIM,
                     hidden_dim=HIDDEN_DIM):
            super().__init__()
            self.state_encoder = StateEncoder(hidden_dim)
            self.perfect_state_encoder = PerfectStateEncoder(hidden_dim)
            self.action_embed = nn.Embedding(
                ACTION_VOCAB_SIZE + 1,
                hidden_dim,
                padding_idx=ACTION_PAD_ID,
            )
            self.history_encoder = RotaryHistoryEncoder(
                self.action_embed,
                hidden_dim=hidden_dim,
                nhead=8,
                num_layers=HISTORY_LAYERS,
                dropout=0.1,
            )
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.perfect_critic = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        def _normalize_inputs(self, state_vec, action_ids, history_rows=None):
            if state_vec.dim() == 1:
                state_vec = state_vec.unsqueeze(0)
            if action_ids.dim() == 1:
                action_ids = action_ids.unsqueeze(0)
            action_ids = action_ids.long()
            if history_rows is None:
                history_rows = torch.zeros(
                    state_vec.shape[0], 1, HISTORY_DIM,
                    dtype=state_vec.dtype,
                    device=state_vec.device)
                history_rows[..., 4] = ACTION_PAD_ID
            elif history_rows.dim() == 1:
                history_rows = history_rows.view(1, 0, HISTORY_DIM)
            elif history_rows.dim() == 2:
                history_rows = history_rows.unsqueeze(0)
            return state_vec, action_ids, history_rows

        def forward(self, state_vec, action_ids, history_rows=None):
            logits, _ = self.forward_actor_critic(
                state_vec, action_ids, history_rows)
            return logits

        def forward_for_seat(self, seat, state_vec, action_ids,
                             history_rows=None):
            return self.forward(state_vec, action_ids, history_rows)

        def forward_actor_critic(self, state_vec, action_ids,
                                 history_rows=None, perfect_state_vec=None):
            state_vec, action_ids, history_rows = self._normalize_inputs(
                state_vec, action_ids, history_rows)
            batch, action_count = action_ids.shape

            state_emb = self.state_encoder(state_vec)
            hist_emb = self.history_encoder(history_rows)
            action_emb = self.action_embed(action_ids.clamp(
                min=0, max=ACTION_PAD_ID))

            state_expand = state_emb.unsqueeze(1).expand(
                batch, action_count, state_emb.shape[-1])
            hist_expand = hist_emb.unsqueeze(1).expand(
                batch, action_count, hist_emb.shape[-1])
            actor_in = torch.cat(
                [state_expand, hist_expand, action_emb], dim=-1)
            logits = self.actor(actor_in).squeeze(-1)

            if perfect_state_vec is not None:
                if perfect_state_vec.dim() == 1:
                    perfect_state_vec = perfect_state_vec.unsqueeze(0)
                perfect_emb = self.perfect_state_encoder(perfect_state_vec)
                value = self.perfect_critic(perfect_emb)
            else:
                value = self.critic(torch.cat([state_emb, hist_emb], dim=-1))
            value = value.squeeze(-1)
            if batch == 1:
                return logits.squeeze(0), value.squeeze(0)
            return logits, value

        def forward_perfect_value(self, perfect_state_vec):
            if perfect_state_vec.dim() == 1:
                perfect_state_vec = perfect_state_vec.unsqueeze(0)
            perfect_emb = self.perfect_state_encoder(perfect_state_vec)
            value = self.perfect_critic(perfect_emb).squeeze(-1)
            return value.squeeze(0) if value.shape[0] == 1 else value
else:
    class CardPolicyNetwork:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError('deep-learning AI requires torch')
