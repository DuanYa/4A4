"""
深度学习AI策略。

策略流程：
1. 通过规则枚举所有合法动作；
2. 编码状态和每个候选动作；
3. 使用共享神经网络模型给候选动作打分；
4. 选择分数最高动作；
5. 若模型不可用或推理失败，自动回退到规则AI，保证线上AI稳定。
"""
from rl.actions import enumerate_legal_actions
from rl.features import encode_state, encode_action, encode_history
from rl.model_store import get_shared_model, DEFAULT_MODEL_NAME
from rl.model import torch


class NeuralPolicy:
    """基于共享神经网络的决策器"""

    def __init__(self, model_name=DEFAULT_MODEL_NAME):
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.model = get_shared_model(self.model_name)

    def choose_play(self, state, hand, level_rank, seat, last_ht,
                    is_free_play, fallback_fn):
        """选择出牌动作，返回牌索引列表；返回None表示pass"""
        if self.model is None or torch is None:
            return fallback_fn()

        actions = enumerate_legal_actions(
            hand, level_rank, last_ht, is_free_play)
        if not actions:
            return fallback_fn()

        try:
            state_vec = torch.tensor(
                encode_state(state, hand, level_rank, seat),
                dtype=torch.float32)
            action_vecs = torch.tensor(
                [encode_action(a, hand, level_rank) for a in actions],
                dtype=torch.float32)
            history_vecs = torch.tensor(
                encode_history(state, seat),
                dtype=torch.float32)
            with torch.no_grad():
                scores = self.model(state_vec, action_vecs, history_vecs)
            best_idx = int(torch.argmax(scores).item())
            best_action = actions[best_idx]
            if best_action['type'] == 'pass':
                return None
            return best_action.get('indices', [])
        except Exception:
            return fallback_fn()
