"""Terminal team rewards for 4A4 self-play training.

The backend AI can only choose card-play actions; cha/dian decisions are still
handled by the existing server path. To avoid teaching selfish play, every
decision made by two teammates receives exactly the same terminal utility.
"""
from models.room import MUST_QUAN_DONG

RANKS = ['3', '4', '5', '6', '7', '8', '9', '10',
         'J', 'Q', 'K', 'A', '2', 'BJ', 'RJ']
STRAIGHT_RANKS = ['3', '4', '5', '6', '7', '8', '9', '10',
                  'J', 'Q', 'K', 'A']


RESULT_LOSE = 'lose'
RESULT_BAN_DONG = 'ban_dong'
RESULT_QUAN_DONG = 'quan_dong'


def _team_orders(finish_order, team):
    return sorted(rank for rank, seat in enumerate(finish_order)
                  if seat % 2 == team)


def team_result(finish_order, team):
    """Return the 4A4 round result for one team from the finish order."""
    orders = _team_orders(finish_order, team)
    if orders == [0, 1]:
        return RESULT_QUAN_DONG
    if orders == [0, 2]:
        return RESULT_BAN_DONG
    return RESULT_LOSE


def round_team_utilities(finish_order, level_rank='3', on_stage_team=0):
    """Compute symmetric terminal utilities for both teams.

    Utility is based only on final team outcome:
    - full hole is better than half hole;
    - special levels 3/J/A only give full upgrade value on full hole, so a
      half hole is intentionally discounted;
    - taking the stage is a small terminal bonus because it changes the
      long-run match state in the existing Room logic.

    The losing team receives the exact negative value, keeping the game
    zero-sum at the team level and preventing reward leakage toward a single
    player's finish order.
    """
    if len(finish_order) < 4:
        return {0: 0.0, 1: 0.0}, {
            'winner_team': -1,
            'winner_result': RESULT_LOSE,
            'stage_change': False,
        }

    results = {team: team_result(finish_order, team) for team in (0, 1)}
    winner_team = -1
    winner_result = RESULT_LOSE
    for team in (0, 1):
        if results[team] != RESULT_LOSE:
            winner_team = team
            winner_result = results[team]
            break

    if winner_team < 0:
        return {0: 0.0, 1: 0.0}, {
            'winner_team': -1,
            'winner_result': RESULT_LOSE,
            'stage_change': False,
        }

    if winner_result == RESULT_QUAN_DONG:
        value = 2.0
    else:
        value = 0.35 if level_rank in MUST_QUAN_DONG else 1.0

    stage_change = winner_team != on_stage_team
    if stage_change:
        value += 0.25

    utilities = {
        winner_team: value,
        1 - winner_team: -value,
    }
    info = {
        'winner_team': winner_team,
        'winner_result': winner_result,
        'stage_change': stage_change,
    }
    return utilities, info


def seat_rewards(finish_order, level_rank='3', on_stage_team=0):
    """Return one terminal reward per seat, shared by teammates."""
    utilities, info = round_team_utilities(
        finish_order, level_rank, on_stage_team)
    rewards = [utilities[seat % 2] for seat in range(4)]
    return rewards, info


def _rank_counts(cards):
    counts = [0] * len(RANKS)
    for card in cards:
        rank = card.rank if hasattr(card, 'rank') else card.get('rank', '3')
        if rank in RANKS:
            counts[RANKS.index(rank)] += 1
    return tuple(counts)


def _remove_long_runs(counts, need, min_len, level_rank):
    counts = list(counts)
    steps = 0
    allowed = [
        RANKS.index(rank) for rank in STRAIGHT_RANKS
        if rank != level_rank
    ]
    while True:
        best = []
        run = []
        prev_pos = None
        for idx in allowed:
            pos = STRAIGHT_RANKS.index(RANKS[idx])
            if counts[idx] >= need and (prev_pos is None or pos == prev_pos + 1):
                run.append(idx)
            else:
                if len(run) > len(best):
                    best = run
                run = [idx] if counts[idx] >= need else []
            prev_pos = pos if counts[idx] >= need else None
        if len(run) > len(best):
            best = run
        if len(best) < min_len:
            break
        for idx in best:
            counts[idx] -= need
        steps += 1
    return tuple(counts), steps


def minimum_play_steps(cards, level_rank='3', _cache=None):
    """Fast perfect-information estimate for minimum play-out steps.

    It ignores opponents' responses and greedily packs long double-straights
    and straights before counting remaining legal groups. This preserves the
    winning-distance signal without putting an expensive search in every actor
    step.
    """
    counts = _rank_counts(cards)
    if _cache is None:
        _cache = {}
    key = (counts, level_rank)
    if key in _cache:
        return _cache[key]

    counts, pair_steps = _remove_long_runs(counts, 2, 3, level_rank)
    counts, straight_steps = _remove_long_runs(counts, 1, 3, level_rank)
    steps = pair_steps + straight_steps

    bj = RANKS.index('BJ')
    rj = RANKS.index('RJ')
    counts = list(counts)
    if counts[bj] and counts[rj]:
        counts[bj] -= 1
        counts[rj] -= 1
        steps += 1

    four = RANKS.index('4')
    ace = RANKS.index('A')
    if counts[four] >= 2 and counts[ace] >= 1:
        counts[four] -= 2
        counts[ace] -= 1
        steps += 1

    for idx, count in enumerate(counts):
        if count <= 0:
            continue
        rank = RANKS[idx]
        if rank in ('BJ', 'RJ'):
            steps += count
        else:
            # Any same-rank residue up to four cards can be played as one
            # single/pair/bomb group in this distance estimate.
            steps += 1
    _cache[key] = steps
    return steps


def team_winning_distances(players, level_rank='3'):
    cache = {}
    seat_steps = {
        player.seat: minimum_play_steps(player.hand, level_rank, cache)
        for player in players
    }
    team_steps = {
        team: min(seat_steps[seat] for seat in seat_steps if seat % 2 == team)
        for team in (0, 1)
    }
    return seat_steps, team_steps


def team_distance_advantages(players, level_rank='3'):
    """Return larger-is-better distance advantage for each team."""
    _, team_steps = team_winning_distances(players, level_rank)
    return {
        0: team_steps[1] - team_steps[0],
        1: team_steps[0] - team_steps[1],
    }


def distance_reward_delta(players_before, players_after, level_rank='3',
                          scale=0.1):
    before = team_distance_advantages(players_before, level_rank)
    after = team_distance_advantages(players_after, level_rank)
    return {
        team: (after[team] - before[team]) * scale
        for team in (0, 1)
    }
