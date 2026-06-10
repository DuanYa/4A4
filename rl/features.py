"""
历史感知特征编码。

编码内容覆盖：
- 自己剩余手牌结构；
- 四名玩家剩余手牌数、队伍、当前轮次；
- 上家牌型与当前阶段；
- 最近若干次出牌/pass/叉/点历史，用于推断各家剩余牌力。
"""
from models.hand_type import HandCategory
from rl.model import STATE_DIM, ACTION_DIM, HISTORY_DIM, HISTORY_LEN

RANKS = ['3', '4', '5', '6', '7', '8', '9', '10',
         'J', 'Q', 'K', 'A', '2', 'BJ', 'RJ']
RANK_TOTALS = {
    '3': 4, '4': 4, '5': 4, '6': 4, '7': 4, '8': 4, '9': 4,
    '10': 4, 'J': 4, 'Q': 4, 'K': 4, 'A': 4, '2': 4,
    'BJ': 1, 'RJ': 1,
}
CATEGORY_ORDER = [
    'SINGLE', 'PAIR', 'STRAIGHT', 'DOUBLE_STRAIGHT',
    'BOMB3', 'BOMB4', 'JOKER_BOMB', 'SI_YAO_SI'
]
ACTION_ORDER = ['play', 'pass', 'cha', 'dian']
PHASE_ORDER = ['waiting', 'dealing', 'playing',
               'cha_asking', 'dian_asking', 'round_end']


def _rank_index(rank):
    try:
        return RANKS.index(rank)
    except ValueError:
        return 0


def _safe_set(vec, idx, value):
    if 0 <= idx < len(vec):
        vec[idx] = value


def _category_name(ht):
    if not ht:
        return ''
    cat = ht.get('category', '') if isinstance(ht, dict) else ht.category
    if isinstance(cat, HandCategory):
        return cat.name
    return str(cat)


def _rank_counts(cards):
    counts = {rank: 0 for rank in RANKS}
    for card in cards:
        rank = card.rank if hasattr(card, 'rank') else card.get('rank', '3')
        if rank in counts:
            counts[rank] += 1
    return counts


def _trailing_pass_count(history):
    count = 0
    for item in reversed(history or []):
        if item.get('action') == 'pass':
            count += 1
            continue
        break
    return count


def _hand_structure(counts):
    pairs = sum(1 for value in counts.values() if value >= 2)
    triples = sum(1 for value in counts.values() if value >= 3)
    quads = sum(1 for value in counts.values() if value >= 4)
    singles = sum(1 for value in counts.values() if value == 1)
    return singles, pairs, triples, quads


def _has_si_yao_si(counts):
    return counts.get('4', 0) >= 2 and counts.get('A', 0) >= 1


def encode_state(state, hand, level_rank, seat):
    """将当前可见局面编码为固定长度状态向量"""
    vec = [0.0] * STATE_DIM

    # 0-14：自己剩余手牌点数分布
    for card in hand:
        vec[_rank_index(card.rank)] += 0.25

    # 15-29：级牌one-hot
    _safe_set(vec, 15 + _rank_index(level_rank), 1.0)

    players = state.get('players', []) if state else []
    player_map = {p.get('seat', 0): p for p in players}

    # 30-45：四家剩余手牌、是否已完成、是否队友、是否自己
    for s in range(4):
        p = player_map.get(s, {})
        base = 30 + s * 4
        vec[base] = p.get('hand_size', 0) / 14.0
        vec[base + 1] = 1.0 if p.get('finished') else 0.0
        vec[base + 2] = 1.0 if seat >= 0 and s % 2 == seat % 2 else 0.0
        vec[base + 3] = 1.0 if s == seat else 0.0

    # 46-53：上家牌型one-hot
    last_ht = state.get('last_hand_type') if state else None
    cat = _category_name(last_ht)
    if cat in CATEGORY_ORDER:
        vec[46 + CATEGORY_ORDER.index(cat)] = 1.0
    if last_ht:
        vec[54] = last_ht.get('key_value', 0) / 20.0
        vec[55] = last_ht.get('length', 0) / 13.0

    # 56-61：阶段one-hot
    phase = state.get('phase', '') if state else ''
    if phase in PHASE_ORDER:
        vec[56 + PHASE_ORDER.index(phase)] = 1.0

    # 62-70：当前控制信息
    vec[62] = seat / 3.0 if seat >= 0 else 0.0
    vec[63] = (seat % 2) if seat >= 0 else 0.0
    vec[64] = 1.0 if state and state.get('is_free_play') else 0.0
    vec[65] = 1.0 if state and state.get('current_player_seat') == seat else 0.0
    vec[66] = state.get('current_player_seat', -1) / 3.0 if state else 0.0
    vec[67] = state.get('last_play_seat', -1) / 3.0 if state else 0.0
    vec[68] = state.get('cha_asking_seat', -1) / 3.0 if state else 0.0
    vec[69] = state.get('dian_asking_seat', -1) / 3.0 if state else 0.0
    vec[70] = state.get('on_stage_team', 0) if state else 0.0

    # 71-85：历史中已明牌出现过的点数计数，辅助推断剩余牌
    for item in (state.get('play_history', []) if state else []):
        for card in item.get('cards', []):
            vec[71 + _rank_index(card.get('rank', '3'))] += 0.25

    history = state.get('play_history', []) if state else []
    hand_counts = _rank_counts(hand)
    played_counts = {rank: 0 for rank in RANKS}
    for item in history:
        for card in item.get('cards', []):
            rank = card.get('rank', '3')
            if rank in played_counts:
                played_counts[rank] += 1

    # 86-95：回合压力、队友/对手手牌压力、台上特殊级别信息
    trailing_passes = _trailing_pass_count(history)
    vec[86] = min(trailing_passes, 3) / 3.0
    hand_sizes = {
        s: player_map.get(s, {}).get('hand_size', 0)
        for s in range(4)
    }
    teammate = (seat + 2) % 4 if seat >= 0 else -1
    opponents = [(seat + 1) % 4, (seat + 3) % 4] if seat >= 0 else []
    vec[87] = hand_sizes.get(teammate, 0) / 14.0
    vec[88] = min([hand_sizes.get(s, 14) for s in opponents] or [14]) / 14.0
    vec[89] = max([hand_sizes.get(s, 0) for s in opponents] or [0]) / 14.0
    vec[90] = len(hand) / 14.0
    on_stage = state.get('on_stage_team', 0) if state else 0
    vec[91] = 1.0 if seat >= 0 and seat % 2 == on_stage else 0.0
    vec[92] = 1.0 if level_rank in ('3', 'J', 'A') else 0.0
    vec[93] = 1.0 if teammate >= 0 and player_map.get(teammate, {}).get('finished') else 0.0
    vec[94] = 1.0 if any(player_map.get(s, {}).get('finished') for s in opponents) else 0.0
    vec[95] = 1.0 if state and state.get('last_play_seat') in opponents else 0.0

    # 96-110：从当前视角估计的未知牌点数余量
    for rank in RANKS:
        idx = _rank_index(rank)
        known = hand_counts.get(rank, 0) + played_counts.get(rank, 0)
        total = RANK_TOTALS.get(rank, 4)
        vec[96 + idx] = max(0, total - known) / float(total)

    # 111-125：自己手牌点数是否可能被叉/点或形成关键牌力
    last_cards = state.get('played_cards', []) if state else []
    last_rank = last_cards[0].get('rank') if last_cards else ''
    if last_rank in RANKS:
        vec[111 + _rank_index(last_rank)] = 1.0
    cha_rank = state.get('cha_rank') if state else None
    if cha_rank in RANKS:
        vec[126] = hand_counts.get(cha_rank, 0) / 4.0
        vec[127] = 1.0 if hand_counts.get(cha_rank, 0) >= 1 else 0.0
        vec[128] = 1.0 if hand_counts.get(cha_rank, 0) >= 2 else 0.0

    # 129-143：自己手牌结构和炸弹资源
    singles, pairs, triples, quads = _hand_structure(hand_counts)
    vec[129] = singles / 14.0
    vec[130] = pairs / 7.0
    vec[131] = triples / 4.0
    vec[132] = quads / 4.0
    vec[133] = hand_counts.get(level_rank, 0) / 4.0
    vec[134] = 1.0 if hand_counts.get('BJ', 0) and hand_counts.get('RJ', 0) else 0.0
    vec[135] = 1.0 if _has_si_yao_si(hand_counts) else 0.0
    vec[136] = 1.0 if any(value >= 3 for rank, value in hand_counts.items()
                          if rank not in ('BJ', 'RJ')) else 0.0
    vec[137] = 1.0 if any(value >= 4 for rank, value in hand_counts.items()
                          if rank not in ('BJ', 'RJ')) else 0.0
    high_ranks = ('A', '2', level_rank, 'BJ', 'RJ')
    vec[138] = sum(hand_counts.get(rank, 0) for rank in set(high_ranks)) / 8.0
    vec[139] = sum(played_counts.get(rank, 0) for rank in ('BJ', 'RJ')) / 2.0
    vec[140] = sum(played_counts.get(rank, 0) for rank in ('4', 'A')) / 8.0
    vec[141] = 1.0 if state and state.get('is_free_play') and len(hand) <= 2 else 0.0
    vec[142] = 1.0 if min([hand_sizes.get(s, 14) for s in opponents] or [14]) <= 2 else 0.0
    vec[143] = 1.0 if hand_sizes.get(teammate, 14) <= 2 else 0.0

    # 144-151：相对座位上的出完顺序，辅助团队协作/接风
    finish_order = state.get('finish_order', []) if state else []
    for order, finished_seat in enumerate(finish_order[:4]):
        if seat >= 0:
            rel = (finished_seat - seat) % 4
            _safe_set(vec, 144 + rel, (4 - order) / 4.0)

    # 152-159：最近行动者的相对座位与是否同队
    for item in reversed(history):
        actor = item.get('seat', -1)
        if actor >= 0 and seat >= 0:
            rel = (actor - seat) % 4
            _safe_set(vec, 152 + rel, 1.0)
            vec[156] = 1.0 if actor == seat else 0.0
            vec[157] = 1.0 if actor % 2 == seat % 2 else 0.0
            vec[158] = 1.0 if actor in opponents else 0.0
            vec[159] = item.get('hand_size_after', 0) / 14.0
            break

    return vec


def encode_history(state, perspective_seat):
    """编码最近HISTORY_LEN条动作历史为Transformer序列"""
    history = (state.get('play_history', []) if state else [])[-HISTORY_LEN:]
    rows = []
    pad = HISTORY_LEN - len(history)
    for _ in range(max(0, pad)):
        rows.append([0.0] * HISTORY_DIM)

    level_rank = state.get('level_rank', '3') if state else '3'
    for item in history:
        vec = [0.0] * HISTORY_DIM
        seat = item.get('seat', -1)
        if seat >= 0:
            vec[seat] = 1.0
            vec[4] = 1.0 if seat == perspective_seat else 0.0
            vec[5] = 1.0 if (perspective_seat >= 0
                              and seat % 2 == perspective_seat % 2) else 0.0
            vec[36] = 1.0 if (perspective_seat >= 0
                              and seat % 2 != perspective_seat % 2) else 0.0
            if perspective_seat >= 0:
                rel = (seat - perspective_seat) % 4
                vec[54 + rel] = 1.0
        action = item.get('action', '')
        if action in ACTION_ORDER:
            vec[6 + ACTION_ORDER.index(action)] = 1.0
        ht = item.get('hand_type')
        cat = _category_name(ht)
        if cat in CATEGORY_ORDER:
            vec[10 + CATEGORY_ORDER.index(cat)] = 1.0
        if ht:
            vec[18] = ht.get('key_value', 0) / 20.0
            vec[19] = ht.get('length', 0) / 13.0
        vec[20] = item.get('hand_size_after', 0) / 14.0
        for card in item.get('cards', []):
            vec[21 + _rank_index(card.get('rank', '3'))] += 0.25
        cards = item.get('cards', [])
        vec[37] = len(cards) / 14.0
        vec[38] = 1.0 if item.get('hand_size_after', 14) <= 2 else 0.0
        vec[39] = 1.0 if any(card.get('rank') in ('BJ', 'RJ')
                              for card in cards) else 0.0
        vec[40] = 1.0 if any(card.get('rank') == level_rank
                              for card in cards) else 0.0
        vec[41] = 1.0 if any(card.get('rank') == '4' for card in cards) else 0.0
        vec[42] = 1.0 if any(card.get('rank') == 'A' for card in cards) else 0.0
        vec[43] = 1.0 if item.get('action') == 'pass' else 0.0
        vec[44] = 1.0 if item.get('action') == 'cha' else 0.0
        vec[45] = 1.0 if item.get('action') == 'dian' else 0.0
        vec[46] = 1.0 if item.get('action') == 'play' else 0.0
        if ht:
            vec[47] = 1.0 if _category_name(ht) in ('BOMB3', 'BOMB4',
                                                     'JOKER_BOMB',
                                                     'SI_YAO_SI') else 0.0
            vec[48] = ht.get('length', 0) / 8.0
        vec[49] = 1.0 if item.get('hand_size_after', 14) == 0 else 0.0
        vec[50] = 1.0 if perspective_seat >= 0 and seat == (perspective_seat + 2) % 4 else 0.0
        vec[51] = 1.0 if perspective_seat >= 0 and seat in (
            (perspective_seat + 1) % 4, (perspective_seat + 3) % 4) else 0.0
        rows.append(vec)
    return rows[-HISTORY_LEN:]


def encode_action(action, hand, level_rank):
    """将候选动作编码为固定长度向量"""
    vec = [0.0] * ACTION_DIM
    action_type = action.get('type')
    if action_type in ('cha', 'dian'):
        do_action = bool(action.get('do'))
        if action_type == 'cha':
            vec[64 if do_action else 65] = 1.0
            need = 2
        else:
            vec[66 if do_action else 67] = 1.0
            need = 1
        rank = action.get('rank') or action.get('cha_rank') or level_rank
        counts = _rank_counts(hand)
        if rank in RANKS:
            vec[13 + _rank_index(rank)] = counts.get(rank, 0) / 4.0
            vec[68] = counts.get(rank, 0) / 4.0
        remain_size = max(0, len(hand) - (need if do_action else 0))
        vec[55] = remain_size / 14.0
        vec[62] = 1.0 if remain_size == 0 else 0.0
        vec[69] = 1.0 if remain_size <= 2 else 0.0
        vec[70] = 1.0 if do_action else 0.0
        vec[71] = 1.0 if not do_action else 0.0
        return vec

    if action.get('type') == 'pass':
        vec[0] = 1.0
        vec[55] = len(hand) / 14.0
        return vec

    indices = action.get('indices', [])
    cards = [hand[i] for i in indices]
    before_counts = _rank_counts(hand)
    ht = action.get('hand_type')
    if ht:
        cat = ht.category.name if isinstance(ht.category, HandCategory) else str(ht.category)
        if cat in CATEGORY_ORDER:
            vec[1 + CATEGORY_ORDER.index(cat)] = 1.0
        vec[10] = ht.key_value / 20.0
        vec[11] = ht.length / 13.0

    vec[12] = len(indices) / 14.0
    for card in cards:
        vec[13 + _rank_index(card.rank)] += 0.25

    vec[28] = 1.0 if len(indices) == len(hand) else 0.0
    vec[29] = sum(card.get_value(level_rank) for card in cards) / 200.0
    vec[30] = max((card.get_value(level_rank) for card in cards), default=0) / 20.0
    vec[31] = min((card.get_value(level_rank) for card in cards), default=0) / 20.0

    # 32-47：动作消耗后自己的剩余手牌点数结构
    remain = list(hand)
    for idx in sorted(indices, reverse=True):
        if 0 <= idx < len(remain):
            remain.pop(idx)
    for card in remain:
        vec[32 + _rank_index(card.rank)] += 0.25
    after_counts = _rank_counts(remain)
    after_singles, after_pairs, after_triples, after_quads = _hand_structure(after_counts)
    vec[48] = 1.0 if any(card.rank == level_rank for card in cards) else 0.0
    vec[49] = 1.0 if any(card.rank in ('BJ', 'RJ') for card in cards) else 0.0
    vec[50] = 1.0 if any(card.rank == '4' for card in cards) else 0.0
    vec[51] = 1.0 if any(card.rank == 'A' for card in cards) else 0.0
    if ht:
        cat_name = ht.category.name if isinstance(ht.category, HandCategory) else str(ht.category)
        vec[52] = 1.0 if cat_name == 'SI_YAO_SI' else 0.0
        vec[53] = 1.0 if cat_name in ('BOMB3', 'BOMB4',
                                       'JOKER_BOMB', 'SI_YAO_SI') else 0.0
        vec[54] = 1.0 if cat_name in ('STRAIGHT', 'DOUBLE_STRAIGHT') else 0.0
    vec[55] = len(remain) / 14.0
    vec[56] = after_singles / 14.0
    vec[57] = after_pairs / 7.0
    vec[58] = after_triples / 4.0
    vec[59] = after_quads / 4.0
    broken_pairs = 0
    cleared_ranks = 0
    for rank in RANKS:
        if before_counts.get(rank, 0) >= 2 and after_counts.get(rank, 0) == 1:
            broken_pairs += 1
        if before_counts.get(rank, 0) > 0 and after_counts.get(rank, 0) == 0:
            cleared_ranks += 1
    vec[60] = broken_pairs / 4.0
    vec[61] = cleared_ranks / 8.0
    vec[62] = 1.0 if len(indices) == len(hand) else 0.0
    vec[63] = (sum(1 for value in after_counts.values() if value >= 3)
               - sum(1 for value in before_counts.values() if value >= 3)) / 4.0
    return vec
