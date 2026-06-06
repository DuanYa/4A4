"""
合法动作枚举器。

神经网络只负责在合法动作之间选择，不直接生成任意牌索引，
这样可以保证深度学习AI不会出非法牌。
"""
from models.hand_type import identify_hand, can_beat, HandCategory
from models import ai_search as search


def _make_play(hand, indices, level_rank):
    cards = [hand[i] for i in indices]
    ht = identify_hand(cards, level_rank)
    if ht is None:
        return None
    return {
        'type': 'play',
        'indices': list(indices),
        'hand_type': ht,
    }


def enumerate_legal_actions(hand, level_rank, last_ht=None,
                            is_free_play=True):
    """枚举当前可执行动作，返回动作字典列表"""
    actions = []
    if not hand:
        return actions

    if is_free_play:
        actions.extend(_free_play_actions(hand, level_rank))
    else:
        actions.append({'type': 'pass', 'indices': [], 'hand_type': None})
        actions.extend(_beat_actions(hand, level_rank, last_ht))

    # 去重：不同搜索路径可能得到同一组索引
    seen = set()
    unique = []
    for action in actions:
        key = (action['type'], tuple(sorted(action.get('indices', []))))
        if key not in seen:
            seen.add(key)
            unique.append(action)
    return unique


def _free_play_actions(hand, level_rank):
    """自由出牌候选动作"""
    proxy = _Proxy(hand, level_rank)
    actions = []

    # 单张、对子、三条、四条是基础动作空间
    for i in range(len(hand)):
        action = _make_play(hand, [i], level_rank)
        if action:
            actions.append(action)

    for getter in (search._get_pairs,
                   lambda p: search._get_n_kind(p, 3),
                   lambda p: search._get_n_kind(p, 4)):
        for indices in getter(proxy):
            action = _make_play(hand, indices, level_rank)
            if action:
                actions.append(action)

    # 单龙：3张起，越长越有战略价值
    max_len = min(len(hand), len(search.STRAIGHT_RANKS))
    for length in range(3, max_len + 1):
        for indices in search._get_straights(proxy, length):
            action = _make_play(hand, indices, level_rank)
            if action:
                actions.append(action)

    # 双龙首出有规则限制：只有一次出完时才允许
    if len(hand) >= 6 and len(hand) % 2 == 0:
        pair_count = len(hand) // 2
        for indices in search._get_dbl_straights(proxy, pair_count):
            if len(indices) == len(hand):
                action = _make_play(hand, indices, level_rank)
                if action:
                    actions.append(action)

    for indices in (search._get_joker_bomb(proxy),
                    search._get_si_yao_si(proxy)):
        if indices:
            action = _make_play(hand, indices, level_rank)
            if action:
                actions.append(action)
    return actions


def _beat_actions(hand, level_rank, last_ht):
    """跟牌候选动作"""
    if last_ht is None:
        return []
    proxy = _Proxy(hand, level_rank)
    actions = []
    cat = last_ht.category

    candidate_indices = []
    if cat == HandCategory.SINGLE:
        candidate_indices.extend([[i] for i in range(len(hand))])
    elif cat == HandCategory.PAIR:
        candidate_indices.extend(search._get_pairs(proxy))
    elif cat == HandCategory.STRAIGHT:
        # 单龙可由更大单龙或任意双龙压制
        candidate_indices.extend(search._get_straights(proxy, last_ht.length))
        max_pairs = len(hand) // 2
        for pair_count in range(3, max_pairs + 1):
            candidate_indices.extend(search._get_dbl_straights(proxy, pair_count))
    elif cat == HandCategory.DOUBLE_STRAIGHT:
        candidate_indices.extend(search._get_dbl_straights(proxy, last_ht.length))

    # 炸弹类动作作为额外兜底候选
    candidate_indices.extend(search._get_n_kind(proxy, 3))
    candidate_indices.extend(search._get_n_kind(proxy, 4))
    jk = search._get_joker_bomb(proxy)
    if jk:
        candidate_indices.append(jk)
    sys = search._get_si_yao_si(proxy)
    if sys:
        candidate_indices.append(sys)

    for indices in candidate_indices:
        action = _make_play(hand, indices, level_rank)
        if action and can_beat(last_ht, action['hand_type']):
            actions.append(action)
    return actions


class _Proxy:
    """复用现有规则搜索函数所需的最小对象"""

    def __init__(self, hand, level_rank):
        self.hand = hand
        self.level_rank = level_rank
