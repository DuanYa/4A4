"""
合法动作枚举器。

神经网络只负责在合法动作之间选择，不直接生成任意牌索引，
这样可以保证深度学习AI不会出非法牌。
"""
from models.hand_type import identify_hand, can_beat, HandCategory
from models import ai_search as search

RANKS = ['3', '4', '5', '6', '7', '8', '9', '10',
         'J', 'Q', 'K', 'A', '2', 'BJ', 'RJ']
REGULAR_RANKS = RANKS[:13]
STRAIGHT_RANKS = list(search.STRAIGHT_RANKS)


def _build_action_vocab():
    specs = []
    specs.append(('pass',))

    for rank in RANKS:
        specs.append(('play', 'SINGLE', (rank,)))
    for rank in REGULAR_RANKS:
        specs.append(('play', 'PAIR', (rank, rank)))
    for rank in REGULAR_RANKS:
        specs.append(('play', 'BOMB3', (rank, rank, rank)))
    for rank in REGULAR_RANKS:
        specs.append(('play', 'BOMB4', (rank, rank, rank, rank)))

    specs.append(('play', 'JOKER_BOMB', ('BJ', 'RJ')))
    specs.append(('play', 'SI_YAO_SI', ('4', '4', 'A')))

    for length in range(3, len(STRAIGHT_RANKS) + 1):
        for start in range(0, len(STRAIGHT_RANKS) - length + 1):
            ranks = tuple(STRAIGHT_RANKS[start:start + length])
            specs.append(('play', 'STRAIGHT', ranks))

    max_pair_count = 7
    for pair_count in range(3, max_pair_count + 1):
        for start in range(0, len(STRAIGHT_RANKS) - pair_count + 1):
            ranks = []
            for rank in STRAIGHT_RANKS[start:start + pair_count]:
                ranks.extend([rank, rank])
            specs.append(('play', 'DOUBLE_STRAIGHT', tuple(ranks)))

    for action_type in ('cha', 'dian'):
        for rank in REGULAR_RANKS:
            specs.append((action_type, True, rank))
            specs.append((action_type, False, rank))

    return tuple(specs)


ACTION_SPECS = _build_action_vocab()
ACTION_TO_ID = {spec: idx for idx, spec in enumerate(ACTION_SPECS)}
ACTION_VOCAB_SIZE = len(ACTION_SPECS)
ACTION_PAD_ID = ACTION_VOCAB_SIZE
PASS_ACTION_ID = ACTION_TO_ID[('pass',)]


def action_spec_to_id(spec):
    return ACTION_TO_ID[spec]


def _rank_sort_key(rank):
    try:
        return RANKS.index(rank)
    except ValueError:
        return len(RANKS)


def _ordered_ranks(cards, category):
    ranks = [card.rank for card in cards]
    if category in ('STRAIGHT', 'DOUBLE_STRAIGHT'):
        return tuple(sorted(ranks, key=lambda rank: STRAIGHT_RANKS.index(rank)))
    return tuple(sorted(ranks, key=_rank_sort_key))


def action_to_spec(action, hand=None, level_rank=None):
    action_type = action.get('type')
    if action_type == 'pass':
        return ('pass',)
    if action_type in ('cha', 'dian'):
        rank = action.get('rank') or action.get('cha_rank') or level_rank
        if rank not in REGULAR_RANKS:
            rank = REGULAR_RANKS[0]
        return (action_type, bool(action.get('do')), rank)
    if action_type != 'play':
        raise KeyError('unsupported action type: %s' % action_type)

    ht = action.get('hand_type')
    if ht is None:
        raise KeyError('play action is missing hand_type')
    category = ht.category.name if hasattr(ht.category, 'name') else str(ht.category)
    indices = action.get('indices', [])
    if hand is None:
        raise KeyError('hand is required for play action ids')
    cards = [hand[i] for i in indices]
    return ('play', category, _ordered_ranks(cards, category))


def action_to_id(action, hand=None, level_rank=None):
    return ACTION_TO_ID[action_to_spec(action, hand, level_rank)]


def actions_to_ids(actions, hand, level_rank):
    return [action_to_id(action, hand, level_rank) for action in actions]


def action_id_to_spec(action_id):
    if action_id < 0 or action_id >= ACTION_VOCAB_SIZE:
        raise KeyError('invalid action id: %s' % action_id)
    return ACTION_SPECS[action_id]


def resolve_action_id(action_id, legal_actions, hand, level_rank):
    """Return a legal action with concrete hand indices for an abstract id."""
    for action in legal_actions:
        try:
            if action_to_id(action, hand, level_rank) == int(action_id):
                return action
        except KeyError:
            continue
    return None


def _iter_index_groups(candidate):
    """Yield one or more index groups from a search result."""
    if not candidate:
        return
    first = candidate[0]
    if isinstance(first, (list, tuple)):
        for indices in candidate:
            yield list(indices)
    else:
        yield list(candidate)


def _add_candidate(target, candidate):
    for indices in _iter_index_groups(candidate):
        target.append(indices)


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


def _semantic_action_key(action, hand):
    try:
        return ('id', action_to_id(action, hand))
    except KeyError:
        action_type = action.get('type')
        indices = sorted(action.get('indices', []))
        ranks = tuple(sorted(hand[i].rank for i in indices))
        return action_type, ranks


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
        key = _semantic_action_key(action, hand)
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

    for candidate in (search._get_joker_bomb(proxy),
                      search._get_si_yao_si(proxy)):
        for indices in _iter_index_groups(candidate):
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
    _add_candidate(candidate_indices, search._get_joker_bomb(proxy))
    _add_candidate(candidate_indices, search._get_si_yao_si(proxy))

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
