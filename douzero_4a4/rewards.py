"""Terminal team rewards for 4A4 self-play training.

The backend AI can only choose card-play actions; cha/dian decisions are still
handled by the existing server path. To avoid teaching selfish play, every
decision made by two teammates receives exactly the same terminal utility.
"""
from models.room import MUST_QUAN_DONG


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

