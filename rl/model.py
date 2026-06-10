"""
History-aware neural policy/value model for 4A4.

The backend loads this class directly, so the training code keeps the online
checkpoint compatible by saving `model_state_dict` for CardPolicyNetwork.
History uses RoPE rotary position encoding and masks zero padding rows.
"""
try:
    import math
    import torch
    import torch.nn as nn
except ImportError:  # Allow rule AI to run on machines without torch.
    math = None
    torch = None
    nn = None

STATE_DIM = 224
ACTION_DIM = 72
HISTORY_DIM = 96
HISTORY_LEN = 80
HIDDEN_DIM = 512
HISTORY_LAYERS = 4


if nn is not None:
    def history_padding_mask(history_vecs):
        """Return True for padded history rows.

        Padding rows are all-zero vectors produced by `encode_history`.
        Completely empty histories are safe: attention is allowed to run, but
        masked pooling below turns the final history embedding into zeros.
        """
        return history_vecs.abs().sum(dim=-1) <= 1e-8


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


    class RotaryHistoryEncoder(nn.Module):
        def __init__(self, history_dim=HISTORY_DIM, hidden_dim=HIDDEN_DIM,
                     nhead=8, num_layers=2, dropout=0.1):
            super().__init__()
            self.proj = nn.Linear(history_dim, hidden_dim)
            self.layers = nn.ModuleList([
                RotaryTransformerBlock(hidden_dim, nhead, dropout)
                for _ in range(num_layers)
            ])
            self.out_norm = nn.LayerNorm(hidden_dim)

        def forward(self, history_vecs):
            if history_vecs.dim() == 2:
                history_vecs = history_vecs.unsqueeze(0)
            padding_mask = history_padding_mask(history_vecs)
            valid = (~padding_mask).to(history_vecs.dtype)
            x = self.proj(history_vecs)
            for layer in self.layers:
                x = layer(x, padding_mask)
            x = self.out_norm(x)
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = x.sum(dim=1) / denom
            return pooled


    class CardPolicyNetwork(nn.Module):
        """Shared symmetric Transformer Q/Actor-Critic model."""

        def __init__(self, state_dim=STATE_DIM,
                     action_dim=ACTION_DIM,
                     history_dim=HISTORY_DIM,
                     hidden_dim=HIDDEN_DIM):
            super().__init__()
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.history_encoder = RotaryHistoryEncoder(
                history_dim=history_dim,
                hidden_dim=hidden_dim,
                nhead=8,
                num_layers=HISTORY_LAYERS,
                dropout=0.1,
            )
            self.action_encoder = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def _normalize_inputs(self, state_vec, action_vecs,
                              history_vecs=None):
            if state_vec.dim() == 1:
                state_vec = state_vec.unsqueeze(0)
            if action_vecs.dim() == 2:
                action_vecs = action_vecs.unsqueeze(0)
            if history_vecs is None:
                history_vecs = torch.zeros(
                    state_vec.shape[0], HISTORY_LEN, HISTORY_DIM,
                    dtype=state_vec.dtype, device=state_vec.device)
            elif history_vecs.dim() == 2:
                history_vecs = history_vecs.unsqueeze(0)
            return state_vec, action_vecs, history_vecs

        def forward(self, state_vec, action_vecs, history_vecs=None):
            logits, _ = self.forward_actor_critic(
                state_vec, action_vecs, history_vecs)
            return logits

        def forward_for_seat(self, seat, state_vec, action_vecs,
                             history_vecs=None):
            # The observation is already encoded from `seat`'s perspective.
            return self.forward(state_vec, action_vecs, history_vecs)

        def forward_actor_critic(self, state_vec, action_vecs,
                                 history_vecs=None):
            state_vec, action_vecs, history_vecs = self._normalize_inputs(
                state_vec, action_vecs, history_vecs)
            batch, action_count, _ = action_vecs.shape

            state_emb = self.state_encoder(state_vec)
            hist_emb = self.history_encoder(history_vecs)
            action_emb = self.action_encoder(action_vecs)

            state_expand = state_emb.unsqueeze(1).expand(
                batch, action_count, state_emb.shape[-1])
            hist_expand = hist_emb.unsqueeze(1).expand(
                batch, action_count, hist_emb.shape[-1])
            actor_in = torch.cat(
                [state_expand, hist_expand, action_emb], dim=-1)
            logits = self.actor(actor_in).squeeze(-1)

            value = self.critic(torch.cat([state_emb, hist_emb], dim=-1))
            value = value.squeeze(-1)
            if batch == 1:
                return logits.squeeze(0), value.squeeze(0)
            return logits, value
else:
    class CardPolicyNetwork:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError('深度学习AI需要先安装 torch')
