"""Feature encoding for the 4A4 reinforcement-learning policy.

The public API intentionally stays small:
- encode_state returns one flat vector whose first 75 values are a
  [5, 15] hand matrix.
- encode_action returns a global action id.
- encode_history returns a variable-length sequence of
  [relative-seat one-hot(4), action-id(1)] rows.
"""
from models.hand_type import HandCategory
from rl.actions import ACTION_PAD_ID, action_to_id
from rl.model import (
    HAND_MATRIX_SIZE,
    HISTORY_DIM,
    PERFECT_CONTEXT_DIM,
    PERFECT_HAND_MATRIX_SIZE,
    PERFECT_STATE_DIM,
    STATE_DIM,
)

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


def _safe_append(values, value):
    values.append(float(value))


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


def _hand_matrix(hand, level_rank):
    counts = _rank_counts(hand)
    matrix = [0.0] * HAND_MATRIX_SIZE
    rank_count = len(RANKS)
    for rank in RANKS:
        rank_idx = _rank_index(rank)
        count = counts.get(rank, 0)
        for channel in range(4):
            if count >= channel + 1:
                matrix[channel * rank_count + rank_idx] = 1.0
        if rank == level_rank:
            matrix[4 * rank_count + rank_idx] = 1.0
    return matrix


def _one_hot(name, order):
    return [1.0 if name == item else 0.0 for item in order]


def encode_state(state, hand, level_rank, seat):
    """Encode visible state. First 75 dims are [5, 15] hand features."""
    vec = _hand_matrix(hand, level_rank)
    context = []

    players = state.get('players', []) if state else []
    player_map = {p.get('seat', 0): p for p in players}
    hand_sizes = {
        s: player_map.get(s, {}).get('hand_size', 0)
        for s in range(4)
    }
    teammate = (seat + 2) % 4 if seat >= 0 else -1
    opponents = [(seat + 1) % 4, (seat + 3) % 4] if seat >= 0 else []

    for s in range(4):
        p = player_map.get(s, {})
        _safe_append(context, p.get('hand_size', 0) / 14.0)
        _safe_append(context, 1.0 if p.get('finished') else 0.0)
        _safe_append(context, 1.0 if seat >= 0 and s % 2 == seat % 2 else 0.0)
        _safe_append(context, 1.0 if s == seat else 0.0)

    last_ht = state.get('last_hand_type') if state else None
    context.extend(_one_hot(_category_name(last_ht), CATEGORY_ORDER))
    if last_ht:
        _safe_append(context, last_ht.get('key_value', 0) / 20.0)
        _safe_append(context, last_ht.get('length', 0) / 13.0)
    else:
        context.extend([0.0, 0.0])

    phase = state.get('phase', '') if state else ''
    context.extend(_one_hot(phase, PHASE_ORDER))
    _safe_append(context, seat / 3.0 if seat >= 0 else 0.0)
    _safe_append(context, (seat % 2) if seat >= 0 else 0.0)
    _safe_append(context, 1.0 if state and state.get('is_free_play') else 0.0)
    _safe_append(context, 1.0 if state and state.get('current_player_seat') == seat else 0.0)
    _safe_append(context, state.get('current_player_seat', -1) / 3.0 if state else 0.0)
    _safe_append(context, state.get('last_play_seat', -1) / 3.0 if state else 0.0)
    _safe_append(context, state.get('cha_asking_seat', -1) / 3.0 if state else 0.0)
    _safe_append(context, state.get('dian_asking_seat', -1) / 3.0 if state else 0.0)
    _safe_append(context, state.get('on_stage_team', 0) if state else 0.0)

    history = state.get('play_history', []) if state else []
    played_counts = {rank: 0 for rank in RANKS}
    for item in history:
        for card in item.get('cards', []):
            rank = card.get('rank', '3')
            if rank in played_counts:
                played_counts[rank] += 1
    for rank in RANKS:
        _safe_append(context, played_counts.get(rank, 0) / max(1, RANK_TOTALS[rank]))

    hand_counts = _rank_counts(hand)
    for rank in RANKS:
        known = hand_counts.get(rank, 0) + played_counts.get(rank, 0)
        total = RANK_TOTALS.get(rank, 4)
        _safe_append(context, max(0, total - known) / float(total))

    _safe_append(context, min(_trailing_pass_count(history), 3) / 3.0)
    _safe_append(context, hand_sizes.get(teammate, 0) / 14.0)
    _safe_append(context, min([hand_sizes.get(s, 14) for s in opponents] or [14]) / 14.0)
    _safe_append(context, max([hand_sizes.get(s, 0) for s in opponents] or [0]) / 14.0)
    _safe_append(context, len(hand) / 14.0)
    on_stage = state.get('on_stage_team', 0) if state else 0
    _safe_append(context, 1.0 if seat >= 0 and seat % 2 == on_stage else 0.0)
    _safe_append(context, 1.0 if teammate >= 0 and player_map.get(teammate, {}).get('finished') else 0.0)
    _safe_append(context, 1.0 if any(player_map.get(s, {}).get('finished') for s in opponents) else 0.0)
    _safe_append(context, 1.0 if state and state.get('last_play_seat') in opponents else 0.0)

    last_cards = state.get('played_cards', []) if state else []
    last_rank = last_cards[0].get('rank') if last_cards else ''
    context.extend(_one_hot(last_rank, RANKS))

    cha_rank = state.get('cha_rank') if state else None
    if cha_rank in RANKS:
        _safe_append(context, hand_counts.get(cha_rank, 0) / 4.0)
        _safe_append(context, 1.0 if hand_counts.get(cha_rank, 0) >= 1 else 0.0)
        _safe_append(context, 1.0 if hand_counts.get(cha_rank, 0) >= 2 else 0.0)
    else:
        context.extend([0.0, 0.0, 0.0])

    singles, pairs, triples, quads = _hand_structure(hand_counts)
    _safe_append(context, singles / 14.0)
    _safe_append(context, pairs / 7.0)
    _safe_append(context, triples / 4.0)
    _safe_append(context, quads / 4.0)
    _safe_append(context, hand_counts.get(level_rank, 0) / 4.0)
    _safe_append(context, 1.0 if hand_counts.get('BJ', 0) and hand_counts.get('RJ', 0) else 0.0)
    _safe_append(context, 1.0 if _has_si_yao_si(hand_counts) else 0.0)
    _safe_append(context, 1.0 if any(value >= 3 for rank, value in hand_counts.items()
                                    if rank not in ('BJ', 'RJ')) else 0.0)
    _safe_append(context, 1.0 if any(value >= 4 for rank, value in hand_counts.items()
                                    if rank not in ('BJ', 'RJ')) else 0.0)
    high_ranks = ('A', '2', level_rank, 'BJ', 'RJ')
    _safe_append(context, sum(hand_counts.get(rank, 0) for rank in set(high_ranks)) / 8.0)
    _safe_append(context, sum(played_counts.get(rank, 0) for rank in ('BJ', 'RJ')) / 2.0)
    _safe_append(context, sum(played_counts.get(rank, 0) for rank in ('4', 'A')) / 8.0)
    _safe_append(context, 1.0 if state and state.get('is_free_play') and len(hand) <= 2 else 0.0)
    _safe_append(context, 1.0 if min([hand_sizes.get(s, 14) for s in opponents] or [14]) <= 2 else 0.0)
    _safe_append(context, 1.0 if hand_sizes.get(teammate, 14) <= 2 else 0.0)

    finish_order = state.get('finish_order', []) if state else []
    finish_signal = [0.0] * 4
    for order, finished_seat in enumerate(finish_order[:4]):
        if seat >= 0:
            rel = (finished_seat - seat) % 4
            finish_signal[rel] = (4 - order) / 4.0
    context.extend(finish_signal)

    recent_actor = [0.0] * 4
    recent_meta = [0.0, 0.0, 0.0, 0.0]
    for item in reversed(history):
        actor = item.get('seat', -1)
        if actor >= 0 and seat >= 0:
            rel = (actor - seat) % 4
            recent_actor[rel] = 1.0
            recent_meta[0] = 1.0 if actor == seat else 0.0
            recent_meta[1] = 1.0 if actor % 2 == seat % 2 else 0.0
            recent_meta[2] = 1.0 if actor in opponents else 0.0
            recent_meta[3] = item.get('hand_size_after', 0) / 14.0
            break
    context.extend(recent_actor)
    context.extend(recent_meta)

    if len(context) > STATE_DIM - HAND_MATRIX_SIZE:
        context = context[:STATE_DIM - HAND_MATRIX_SIZE]
    if len(context) < STATE_DIM - HAND_MATRIX_SIZE:
        context.extend([0.0] * (STATE_DIM - HAND_MATRIX_SIZE - len(context)))
    vec.extend(context)
    return vec


def encode_perfect_state(game, perspective_seat, visible_state=None):
    """Encode full training-only state with all four hands.

    This mirrors PerfectDou's perfect-training/imperfect-execution idea:
    the vector is stored only in replay transitions for the teacher loss and
    is never required by the online policy.
    """
    level_rank = getattr(game, 'level_rank', '3')
    vec = []
    for rel in range(4):
        seat = (perspective_seat + rel) % 4 if perspective_seat >= 0 else rel
        player = game.players[seat]
        vec.extend(_hand_matrix(player.hand, level_rank))
    if len(vec) > PERFECT_HAND_MATRIX_SIZE:
        vec = vec[:PERFECT_HAND_MATRIX_SIZE]
    if len(vec) < PERFECT_HAND_MATRIX_SIZE:
        vec.extend([0.0] * (PERFECT_HAND_MATRIX_SIZE - len(vec)))

    state = visible_state if visible_state is not None else game.get_state(
        for_seat=perspective_seat)
    context = []
    for rel in range(4):
        seat = (perspective_seat + rel) % 4 if perspective_seat >= 0 else rel
        player = game.players[seat]
        _safe_append(context, player.hand_size() / 14.0)
        _safe_append(context, 1.0 if player.finished else 0.0)
        _safe_append(context, 1.0 if seat % 2 == perspective_seat % 2 else 0.0)
        _safe_append(context, 1.0 if seat == perspective_seat else 0.0)

    phase = state.get('phase', '') if state else ''
    context.extend(_one_hot(phase, PHASE_ORDER))
    last_ht = state.get('last_hand_type') if state else None
    context.extend(_one_hot(_category_name(last_ht), CATEGORY_ORDER))
    if last_ht:
        _safe_append(context, last_ht.get('key_value', 0) / 20.0)
        _safe_append(context, last_ht.get('length', 0) / 13.0)
    else:
        context.extend([0.0, 0.0])

    _safe_append(context, perspective_seat / 3.0 if perspective_seat >= 0 else 0.0)
    _safe_append(context, getattr(game, 'current_player_seat', -1) / 3.0)
    _safe_append(context, getattr(game, 'last_play_seat', -1) / 3.0)
    _safe_append(context, 1.0 if getattr(game, 'is_free_play', False) else 0.0)
    _safe_append(context, getattr(game, 'on_stage_team', 0))
    _safe_append(context, getattr(game, 'cha_asking_seat', -1) / 3.0)
    _safe_append(context, getattr(game, 'dian_asking_seat', -1) / 3.0)
    _safe_append(context, getattr(game, 'cha_player_seat', -1) / 3.0)
    _safe_append(context, getattr(game, 'pass_count', 0) / 3.0)

    played_by_rel = [[0.0] * len(RANKS) for _ in range(4)]
    for item in getattr(game, 'play_history', []):
        actor = item.get('seat', -1)
        if actor < 0 or perspective_seat < 0:
            continue
        rel = (actor - perspective_seat) % 4
        for card in item.get('cards', []):
            rank = card.get('rank', '3')
            if rank in RANKS:
                played_by_rel[rel][_rank_index(rank)] += (
                    1.0 / max(1, RANK_TOTALS[rank]))
    for rel in range(4):
        context.extend(played_by_rel[rel])

    finish_signal = [0.0] * 4
    for order, finished_seat in enumerate(getattr(game, 'finish_order', [])[:4]):
        if perspective_seat >= 0:
            rel = (finished_seat - perspective_seat) % 4
            finish_signal[rel] = (4 - order) / 4.0
    context.extend(finish_signal)

    if len(context) > PERFECT_CONTEXT_DIM:
        context = context[:PERFECT_CONTEXT_DIM]
    if len(context) < PERFECT_CONTEXT_DIM:
        context.extend([0.0] * (PERFECT_CONTEXT_DIM - len(context)))
    vec.extend(context)
    if len(vec) != PERFECT_STATE_DIM:
        raise ValueError('perfect state dim mismatch: %d' % len(vec))
    return vec


def _history_action_id(item, level_rank):
    action = item.get('action', '')
    if action == 'pass':
        return action_to_id({'type': 'pass'})
    if action in ('cha', 'dian'):
        cards = item.get('cards', [])
        rank = cards[0].get('rank', level_rank) if cards else level_rank
        return action_to_id({'type': action, 'do': True, 'rank': rank})
    if action == 'play':
        cards = item.get('cards', [])
        hand_type = item.get('hand_type')
        fake_hand = [
            type('HistoryCard', (), {
                'rank': card.get('rank', '3'),
                'suit': card.get('suit', ''),
            })()
            for card in cards
        ]
        fake_action = {
            'type': 'play',
            'indices': list(range(len(fake_hand))),
            'hand_type': type('HistoryHandType', (), {
                'category': type('HistoryCategory', (), {
                    'name': _category_name(hand_type),
                })(),
            })(),
        }
        return action_to_id(fake_action, fake_hand, level_rank)
    return ACTION_PAD_ID


def encode_history(state, perspective_seat):
    """Return variable-length history rows: 4 relative-seat bits + action id."""
    history = state.get('play_history', []) if state else []
    level_rank = state.get('level_rank', '3') if state else '3'
    rows = []
    for item in history:
        action = item.get('action', '')
        if action in ('cha_pass', 'cha_all_pass', 'dian_pass', 'dian_all_pass'):
            continue
        if action in ('cha', 'dian') and not item.get('cards'):
            continue
        seat = item.get('seat', -1)
        rel = [0.0] * 4
        if seat >= 0 and perspective_seat >= 0:
            rel[(seat - perspective_seat) % 4] = 1.0
        try:
            action_id = _history_action_id(item, level_rank)
        except KeyError:
            continue
        if action_id == ACTION_PAD_ID:
            continue
        rows.append(rel + [float(action_id)])
    return rows


def encode_action(action, hand, level_rank):
    """Encode a legal action as one global action id."""
    return action_to_id(action, hand, level_rank)
