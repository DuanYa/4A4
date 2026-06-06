"""
历史感知 Actor-Critic 神经网络。

四幺四不是只看当前能否管牌的单步问题：
- 每名玩家过去出过什么牌会影响剩余牌推断；
- 自己剩余手牌结构决定是否要拆牌、保炸、送队友；
- 出牌动作数量动态变化，需要在合法候选动作上打分。

因此模型采用：
1. 状态MLP：编码当前可见全局状态；
2. 历史Transformer：编码最近若干次出牌/pass/叉/点事件；
3. 动作MLP：编码每个候选动作；
4. Actor头：对每个合法动作输出logit；
5. Critic头：输出当前状态价值，用于PPO稳定训练。
"""
try:
    import torch
    import torch.nn as nn
except ImportError:  # 允许未安装torch时仍能启动普通规则AI
    torch = None
    nn = None

STATE_DIM = 224
ACTION_DIM = 64
HISTORY_DIM = 96
HISTORY_LEN = 40
HIDDEN_DIM = 512


if nn is not None:
    class CardPolicyNetwork(nn.Module):
        """兼容旧名称的历史感知Actor-Critic网络"""

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
            self.history_proj = nn.Linear(history_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True,
                activation='gelu',
            )
            self.history_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=2)
            self.action_encoder = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
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
            """返回每个合法动作的logit，兼容旧推理接口"""
            logits, _ = self.forward_actor_critic(
                state_vec, action_vecs, history_vecs)
            return logits

        def forward_actor_critic(self, state_vec, action_vecs,
                                 history_vecs=None):
            """同时输出动作logits和状态价值"""
            state_vec, action_vecs, history_vecs = self._normalize_inputs(
                state_vec, action_vecs, history_vecs)
            batch, action_count, _ = action_vecs.shape

            state_emb = self.state_encoder(state_vec)
            hist_tokens = self.history_proj(history_vecs)
            hist_encoded = self.history_encoder(hist_tokens)
            hist_emb = hist_encoded.mean(dim=1)
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
        """torch未安装时的占位类，避免普通AI导入失败"""
        def __init__(self, *args, **kwargs):
            raise ImportError('深度学习AI需要先安装 torch')
