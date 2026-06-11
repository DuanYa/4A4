"""DouZero-style Deep Monte Carlo trainer for 4A4.

The saved checkpoint is intentionally compatible with the existing backend:
`rl.model_store` can load it as a normal `CardPolicyNetwork` checkpoint without
any server-side code changes.
"""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, deque
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from douzero_4a4.rewards import seat_rewards
from models.game import Game, GamePhase
from models.hand_type import HandType, HandCategory, identify_hand
from models.player import Player
from models.room import LEVEL_ORDER
from rl.actions import enumerate_legal_actions
from rl.features import encode_action, encode_history, encode_state
from rl.model import ACTION_PAD_ID, HISTORY_DIM, CardPolicyNetwork, torch
from models import ai_search


@dataclass
class Decision:
    state_vec: object
    history_vec: object
    action_ids: object
    action_index: int
    seat: int
    reward: float = 0.0


class RuleProxy:
    def __init__(self, hand, level_rank):
        self.hand = hand
        self.level_rank = level_rank


def _dict_to_hand_type(data):
    if not data:
        return None
    return HandType(
        HandCategory[data['category']],
        [],
        data.get('key_value', 0),
        data.get('length', 0),
    )


def _rule_action(hand, level_rank, last_ht, is_free_play):
    proxy = RuleProxy(hand, level_rank)
    if is_free_play:
        return ai_search._find_free_play(proxy)
    return ai_search._find_beat_play(proxy, last_ht)


def _training_should_cha(game, seat):
    player = game.players[seat]
    own_after = max(0, player.hand_size() - 2)
    source = game.cha_source_seat
    if own_after <= 2:
        return True
    if source >= 0 and source % 2 == seat % 2:
        if game.players[source].hand_size() <= 2:
            return False
    return True


def _training_should_dian(game, seat):
    player = game.players[seat]
    own_after = max(0, player.hand_size() - 1)
    cha_player = game.cha_player_seat
    if own_after <= 2:
        return True
    if cha_player >= 0 and cha_player % 2 == seat % 2:
        if game.players[cha_player].hand_size() <= 2:
            return False
    return True


def _forward_action_values(model, seat, state_vec, action_vecs, history_vec):
    if hasattr(model, 'forward_for_seat'):
        return model.forward_for_seat(
            seat, state_vec, action_vecs, history_vec)
    return model(state_vec, action_vecs, history_vec)


def _history_tensor(state, seat, device):
    rows = encode_history(state, seat)
    if not rows:
        return torch.empty((0, HISTORY_DIM), dtype=torch.float32, device=device)
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _pad_history_tensors(items, device):
    max_len = max([item.shape[0] for item in items] + [1])
    batch = torch.zeros(
        len(items), max_len, HISTORY_DIM,
        dtype=torch.float32, device=device)
    batch[..., 4] = ACTION_PAD_ID
    for row, item in enumerate(items):
        if item.shape[0] > 0:
            batch[row, :item.shape[0]] = item.to(device, non_blocking=True)
    return batch


def _select_from_actions(model, state, hand, level_rank, seat, actions,
                         epsilon, device, train=True):
    if not actions:
        return {'type': 'pass', 'indices': [], 'hand_type': None}, None, {}

    state_vec = torch.tensor(
        encode_state(state, hand, level_rank, seat),
        dtype=torch.float32, device=device)
    history_vec = _history_tensor(state, seat, device)
    action_ids = torch.tensor(
        [encode_action(action, hand, level_rank) for action in actions],
        dtype=torch.long, device=device)

    with torch.no_grad():
        q_values = _forward_action_values(
            model, seat, state_vec, action_ids, history_vec)
        greedy_idx = int(torch.argmax(q_values).item())

    if train and random.random() < epsilon:
        action_idx = random.randrange(len(actions))
        explored = True
    else:
        action_idx = greedy_idx
        explored = False

    decision = None
    if train:
        decision = Decision(
            state_vec=state_vec.detach(),
            history_vec=history_vec.detach(),
            action_ids=action_ids.detach(),
            action_index=action_idx,
            seat=seat,
        )
    info = {
        'candidate_count': len(actions),
        'explored': explored,
        'selected_q': float(q_values[action_idx].detach().cpu().item()),
        'greedy_q': float(q_values[greedy_idx].detach().cpu().item()),
    }
    return actions[action_idx], decision, info


def _select_model_action(model, state, hand, level_rank, seat, last_ht,
                         is_free_play, epsilon, device, train=True):
    actions = enumerate_legal_actions(
        hand, level_rank, last_ht, is_free_play)
    return _select_from_actions(
        model, state, hand, level_rank, seat, actions, epsilon, device, train)


def _select_binary_action(model, state, hand, level_rank, seat, action_type,
                          rank, epsilon, device, train=True):
    actions = [
        {'type': action_type, 'do': True, 'rank': rank},
        {'type': action_type, 'do': False, 'rank': rank},
    ]
    return _select_from_actions(
        model, state, hand, level_rank, seat, actions, epsilon, device, train)


def _random_level(configured):
    if configured and configured != 'random':
        return configured
    return random.choice(LEVEL_ORDER)


def run_episode(model, args, device, train=True, model_team=None,
                return_record=False):
    level_rank = _random_level(args.level_rank)
    on_stage_team = random.randrange(2)
    players = [Player('dz4a4-%d' % i, 'DZ4A4-%d' % i, i)
               for i in range(4)]
    game = Game(players, level_rank, on_stage_team)
    game.start()
    record = None
    if return_record:
        record = {
            'level_rank': level_rank,
            'on_stage_team': on_stage_team,
            'model_team': model_team,
            'initial_hands': {
                str(p.seat): [card.to_dict() for card in p.hand]
                for p in players
            },
        }

    trajectories = [[] for _ in range(4)]
    stats = Counter()
    stats['level_' + level_rank] += 1
    stats['on_stage_%d' % on_stage_team] += 1
    candidate_counts = []
    q_selected = []
    q_greedy = []
    epsilon = args.epsilon_eval
    if train:
        epsilon = args.epsilon

    for _ in range(args.max_episode_steps):
        if game.phase == GamePhase.ROUND_END:
            break

        if game.phase == GamePhase.CHA_ASKING:
            stats['cha_asks'] += 1
            seat = game.cha_asking_seat
            player = players[seat]
            state = game.get_state(for_seat=seat)
            use_model = model_team is None or seat % 2 == model_team
            if use_model:
                action, decision, info = _select_binary_action(
                    model, state, player.hand, level_rank, seat,
                    'cha', game.cha_rank, epsilon, device, train)
                if decision is not None:
                    trajectories[seat].append(decision)
                if info:
                    candidate_counts.append(info['candidate_count'])
                    q_selected.append(info['selected_q'])
                    q_greedy.append(info['greedy_q'])
                    if info['explored']:
                        stats['explore_actions'] += 1
                do_cha = bool(action.get('do'))
            else:
                do_cha = _training_should_cha(game, seat)
            if not do_cha:
                stats['cha_declines'] += 1
            game.respond_cha(seat, do_cha)
            continue

        if game.phase == GamePhase.DIAN_ASKING:
            stats['dian_asks'] += 1
            seat = game.dian_asking_seat
            player = players[seat]
            state = game.get_state(for_seat=seat)
            use_model = model_team is None or seat % 2 == model_team
            if use_model:
                action, decision, info = _select_binary_action(
                    model, state, player.hand, level_rank, seat,
                    'dian', game.cha_rank, epsilon, device, train)
                if decision is not None:
                    trajectories[seat].append(decision)
                if info:
                    candidate_counts.append(info['candidate_count'])
                    q_selected.append(info['selected_q'])
                    q_greedy.append(info['greedy_q'])
                    if info['explored']:
                        stats['explore_actions'] += 1
                do_dian = bool(action.get('do'))
            else:
                do_dian = _training_should_dian(game, seat)
            if not do_dian:
                stats['dian_declines'] += 1
            game.respond_dian(seat, do_dian)
            continue

        if game.phase != GamePhase.PLAYING:
            stats['idle_steps'] += 1
            continue

        seat = game.current_player_seat
        player = players[seat]
        state = game.get_state(for_seat=seat)
        last_ht = _dict_to_hand_type(state.get('last_hand_type'))
        use_model = model_team is None or seat % 2 == model_team

        if use_model:
            action, decision, info = _select_model_action(
                model, state, player.hand, level_rank, seat, last_ht,
                state.get('is_free_play', False), epsilon, device, train)
            if decision is not None:
                trajectories[seat].append(decision)
            if info:
                candidate_counts.append(info['candidate_count'])
                q_selected.append(info['selected_q'])
                q_greedy.append(info['greedy_q'])
                if info['explored']:
                    stats['explore_actions'] += 1
        else:
            indices = _rule_action(
                player.hand, level_rank, last_ht,
                state.get('is_free_play', False))
            if indices is None:
                action = {'type': 'pass', 'indices': [], 'hand_type': None}
            else:
                cards = [player.hand[i] for i in indices]
                action = {
                    'type': 'play',
                    'indices': indices,
                    'hand_type': identify_hand(cards, level_rank),
                }

        if action['type'] == 'pass':
            stats['passes'] += 1
            result = game.player_pass(seat)
        else:
            stats['plays'] += 1
            hand_type = action.get('hand_type')
            if hand_type:
                stats['hand_' + hand_type.category.name] += 1
            result = game.play_cards(seat, action['indices'])

        if not result.get('success'):
            stats['invalid'] += 1
            if not state.get('is_free_play', False):
                game.player_pass(seat)

    finished = game.phase == GamePhase.ROUND_END
    rewards, reward_info = seat_rewards(
        game.finish_order, level_rank, on_stage_team)
    if not finished:
        rewards = [0.0, 0.0, 0.0, 0.0]
        reward_info = {
            'winner_team': -1,
            'winner_result': 'truncated',
            'stage_change': False,
        }

    transitions = []
    for seat, decisions in enumerate(trajectories):
        for decision in decisions:
            decision.reward = rewards[seat]
            transitions.append(decision)

    stats['finished'] += int(finished)
    stats['winner_team_%s' % reward_info['winner_team']] += 1
    stats['result_' + reward_info['winner_result']] += 1
    winner_team = reward_info['winner_team']
    winner_result = reward_info['winner_result']
    if winner_team in (0, 1):
        stats['team%d_wins' % winner_team] += 1
        stats['team%d_%s' % (winner_team, winner_result)] += 1
    stats['stage_changes'] += int(reward_info['stage_change'])
    stats['transitions'] += len(transitions)
    stats['team0_reward_x1000'] += int(rewards[0] * 1000)
    stats['team1_reward_x1000'] += int(rewards[1] * 1000)
    stats['avg_candidates_x1000'] += int(
        (sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0) * 1000)
    stats['avg_selected_q_x1000'] += int(
        (sum(q_selected) / len(q_selected) if q_selected else 0) * 1000)
    stats['avg_greedy_q_x1000'] += int(
        (sum(q_greedy) / len(q_greedy) if q_greedy else 0) * 1000)
    if return_record:
        record.update({
            'finished': finished,
            'finish_order': list(game.finish_order),
            'winner_team': reward_info['winner_team'],
            'winner_result': reward_info['winner_result'],
            'stage_change': reward_info['stage_change'],
            'rewards': list(rewards),
            'play_history': list(game.play_history),
            'stats': dict(stats),
        })
        return transitions, stats, record
    return transitions, stats


def dmc_update(model, optimizer, transitions, args, device):
    if not transitions:
        return {'updated': False}
    random.shuffle(transitions)
    losses = []
    abs_errors = []
    q_means = []
    grad_norm = 0.0
    batches = 0

    for start in range(0, len(transitions), args.batch_size):
        batch = transitions[start:start + args.batch_size]
        max_actions = max(item.action_ids.shape[0] for item in batch)
        state = torch.stack(
            [item.state_vec for item in batch]).to(device, non_blocking=True)
        history = _pad_history_tensors(
            [item.history_vec for item in batch], device)
        actions = torch.full(
            (len(batch), max_actions),
            ACTION_PAD_ID,
            dtype=torch.long, device=device)
        for row, item in enumerate(batch):
            count = item.action_ids.shape[0]
            actions[row, :count] = item.action_ids.to(
                device, non_blocking=True)
        action_index = torch.tensor(
            [item.action_index for item in batch],
            dtype=torch.long, device=device)
        target = torch.tensor(
            [item.reward for item in batch],
            dtype=torch.float32, device=device)

        q_values = model(state, actions, history)
        pred = q_values.gather(1, action_index.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm)
        optimizer.step()
        batches += 1

        with torch.no_grad():
            losses.append(float(loss.detach().cpu().item()))
            abs_errors.append(float(torch.mean(torch.abs(pred - target)).cpu().item()))
            q_means.append(float(torch.mean(pred).detach().cpu().item()))
            grad_norm = float(grad_norm_tensor)

    return {
        'updated': True,
        'loss': sum(losses) / len(losses),
        'abs_error': sum(abs_errors) / len(abs_errors),
        'q_mean': sum(q_means) / len(q_means),
        'grad_norm': grad_norm,
        'update_batches_done': batches,
    }


def save_checkpoint(model, optimizer, path, frames, episodes, args, metrics):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': frames,
        'episode': episodes,
        'model_type': 'shared_rope_transformer_full_action_dmc',
        'backend_model_class': 'rl.model.CardPolicyNetwork',
        'reward_version': 'terminal_team_utility_v1',
        'args': vars(args),
        'metrics': metrics,
    }, path)


def _checkpoint_path(base_path, frames):
    root, ext = os.path.splitext(base_path)
    return '%s_step_%d%s' % (root, frames, ext or '.pt')


def _load_compatible(module, state_dict):
    current = module.state_dict()
    compatible = {
        key: value for key, value in state_dict.items()
        if key in current and current[key].shape == value.shape
    }
    current.update(compatible)
    module.load_state_dict(current)


def _load_resume(model, optimizer, path, device):
    if not path or not os.path.exists(path):
        return 0, 0
    payload = torch.load(path, map_location=device)
    if 'model_state_dict' in payload:
        _load_compatible(model, payload['model_state_dict'])
    if 'optimizer_state_dict' in payload:
        try:
            optimizer.load_state_dict(payload['optimizer_state_dict'])
        except ValueError:
            pass
    return int(payload.get('global_step', 0)), int(payload.get('episode', 0))


def _counter_rate(counter, key, total_key='episodes'):
    total = max(1, counter[total_key])
    return counter[key] / total


def evaluate(model, args, device, metric_prefix='eval'):
    total = Counter()
    for i in range(args.eval_episodes):
        model_team = i % 2
        _, stats = run_episode(
            model, args, device, train=False, model_team=model_team)
        total.update(stats)
        total['episodes'] += 1
        if stats.get('winner_team_%d' % model_team):
            total['model_team_wins'] += 1
            if stats.get('result_quan_dong'):
                total['model_team_quan_dong'] += 1
            if stats.get('result_ban_dong'):
                total['model_team_ban_dong'] += 1
        else:
            rule_team = 1 - model_team
            if stats.get('winner_team_%d' % rule_team):
                total['rule_team_wins'] += 1
                if stats.get('result_quan_dong'):
                    total['rule_team_quan_dong'] += 1
                if stats.get('result_ban_dong'):
                    total['rule_team_ban_dong'] += 1
    return {
        metric_prefix + '_episodes': total['episodes'],
        metric_prefix + '_model_team_win_rate': _counter_rate(
            total, 'model_team_wins'),
        metric_prefix + '_model_team_full_hole_rate': _counter_rate(
            total, 'model_team_quan_dong'),
        metric_prefix + '_model_team_half_hole_rate': _counter_rate(
            total, 'model_team_ban_dong'),
        metric_prefix + '_rule_team_win_rate': _counter_rate(
            total, 'rule_team_wins'),
        metric_prefix + '_rule_team_full_hole_rate': _counter_rate(
            total, 'rule_team_quan_dong'),
        metric_prefix + '_rule_team_half_hole_rate': _counter_rate(
            total, 'rule_team_ban_dong'),
        metric_prefix + '_full_hole_rate': _counter_rate(
            total, 'result_quan_dong'),
        metric_prefix + '_half_hole_rate': _counter_rate(
            total, 'result_ban_dong'),
        metric_prefix + '_truncated_rate': _counter_rate(
            total, 'result_truncated'),
        metric_prefix + '_invalid': total['invalid'],
    }


def write_eval_records(model, args, device, path, frames, episodes):
    if not path or getattr(args, 'eval_record_episodes', 0) <= 0:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    records = []
    for i in range(args.eval_record_episodes):
        model_team = i % 2
        _, _, record = run_episode(
            model, args, device, train=False, model_team=model_team,
            return_record=True)
        record['frames'] = frames
        record['episodes'] = episodes
        record['record_index'] = i
        records.append(record)
    with open(path, 'a', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def _format_float(value):
    if value is None:
        return 'NA'
    return '%.5f' % value


CN_KEYS = {
    'episode': '局数',
    'frames': '样本步数',
    'fps': '每秒样本',
    'epsilon': '探索率',
    'buffer': '缓冲区样本',
    'loss': '训练损失',
    'abs_error': '平均绝对误差',
    'q_mean': '平均Q值',
    'grad_norm': '梯度范数',
    'update_batches_done': '更新批次数',
    'team0_reward_mean': 'A队平均终局收益',
    'terminal_utility_abs_mean': '终局收益绝对值',
    'finish_rate': '正常结束率',
    'full_hole_rate': '全洞率',
    'half_hole_rate': '半洞率',
    'stage_change_rate': '台上切换率',
    'team0_win_rate': '0队胜率',
    'team1_win_rate': '1队胜率',
    'team0_full_hole_rate': '0队全洞率',
    'team0_half_hole_rate': '0队半洞率',
    'team1_full_hole_rate': '1队全洞率',
    'team1_half_hole_rate': '1队半洞率',
    'invalid': '非法动作数',
    'avg_candidates': '平均候选动作数',
    'avg_selected_q': '平均所选Q值',
    'avg_greedy_q': '平均贪心Q值',
    'avg_q_gap': '平均探索Q差',
    'plays': '出牌次数',
    'passes': '过牌次数',
    'pass_rate': '过牌率',
    'explore_actions': '探索动作数',
    'explore_rate': '探索动作率',
    'cha_asks': '叉牌询问数',
    'dian_asks': '点牌询问数',
    'cha_declines': '不叉次数',
    'dian_declines': '不点次数',
    'cha_decline_rate': '不叉率',
    'dian_decline_rate': '不点率',
    'eval_model_team_win_rate': '评估胜率',
    'eval_model_team_full_hole_rate': '评估模型队全洞率',
    'eval_model_team_half_hole_rate': '评估模型队半洞率',
    'eval_rule_team_win_rate': '评估规则队胜率',
    'eval_rule_team_full_hole_rate': '评估规则队全洞率',
    'eval_rule_team_half_hole_rate': '评估规则队半洞率',
    'eval_full_hole_rate': '评估全洞率',
    'eval_half_hole_rate': '评估半洞率',
    'eval_truncated_rate': '评估截断率',
    'eval_invalid': '评估非法动作数',
}


def _log_line(metrics):
    keys = [
        'episode', 'frames', 'fps', 'epsilon', 'buffer',
        'loss', 'abs_error', 'q_mean', 'grad_norm',
        'update_batches_done',
        'team0_reward_mean', 'terminal_utility_abs_mean',
        'finish_rate', 'full_hole_rate', 'half_hole_rate',
        'stage_change_rate', 'team0_win_rate', 'team1_win_rate',
        'team0_full_hole_rate', 'team0_half_hole_rate',
        'team1_full_hole_rate', 'team1_half_hole_rate', 'invalid',
        'avg_candidates', 'avg_selected_q', 'avg_greedy_q',
        'avg_q_gap', 'plays', 'passes', 'pass_rate',
        'explore_actions', 'explore_rate',
        'cha_asks', 'cha_declines', 'cha_decline_rate',
        'dian_asks', 'dian_declines', 'dian_decline_rate',
        'eval_model_team_win_rate', 'eval_model_team_full_hole_rate',
        'eval_model_team_half_hole_rate', 'eval_rule_team_win_rate',
        'eval_rule_team_full_hole_rate', 'eval_rule_team_half_hole_rate',
        'eval_full_hole_rate', 'eval_half_hole_rate',
        'eval_truncated_rate', 'eval_invalid',
    ]
    parts = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            value = _format_float(value)
        parts.append('%s=%s' % (CN_KEYS.get(key, key), value))
    print(' '.join(parts), flush=True)


def _write_jsonl(path, metrics):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    cn_metrics = {CN_KEYS.get(key, key): value for key, value in metrics.items()}
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(cn_metrics, ensure_ascii=False, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=200000)
    parser.add_argument('--episodes', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--update-frames', type=int, default=4096)
    parser.add_argument('--epsilon', type=float, default=0.08)
    parser.add_argument('--epsilon-eval', type=float, default=0.0)
    parser.add_argument('--max-grad-norm', type=float, default=40.0)
    parser.add_argument('--max-episode-steps', type=int, default=700)
    parser.add_argument('--num-actors', type=int, default=1)
    parser.add_argument('--collect-episodes', type=int, default=1)
    parser.add_argument('--torch-threads', type=int, default=0)
    parser.add_argument('--level-rank', default='random',
                        choices=['random'] + LEVEL_ORDER)
    parser.add_argument('--save', default=os.path.join(
        ROOT, 'checkpoints', 'douzero_4a4.pt'))
    parser.add_argument('--resume', default='')
    parser.add_argument('--checkpoint-every', type=int, default=10000)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--log-jsonl', default=os.path.join(
        ROOT, 'logs', 'douzero_4a4_metrics.jsonl'))
    parser.add_argument('--eval-every', type=int, default=50000)
    parser.add_argument('--eval-episodes', type=int, default=64)
    parser.add_argument('--eval-record-episodes', type=int, default=4)
    parser.add_argument('--eval-record-jsonl', default=os.path.join(
        ROOT, 'logs', 'douzero_4a4_eval_records.jsonl'))
    args = parser.parse_args()

    if torch is None:
        raise RuntimeError('训练需要 torch，请在远端执行: conda activate torch')
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)

    model = CardPolicyNetwork().to(device)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.lr, momentum=0.0,
        eps=1e-5, alpha=0.99)
    frames, episode = _load_resume(model, optimizer, args.resume, device)
    buffer = []
    next_checkpoint = None
    if args.checkpoint_every:
        next_checkpoint = (
            (frames // args.checkpoint_every) + 1) * args.checkpoint_every
    next_eval = None
    if args.eval_every:
        next_eval = ((frames // args.eval_every) + 1) * args.eval_every
    recent = Counter()
    recent_team0_rewards = deque(maxlen=200)
    recent_abs_utilities = deque(maxlen=200)
    start_time = time.time()
    last_update = {'updated': False}

    while True:
        if args.episodes and episode >= args.episodes:
            break
        if args.frames and frames >= args.frames:
            break

        collect_count = max(1, args.collect_episodes, args.num_actors)
        if args.episodes:
            collect_count = min(collect_count, args.episodes - episode)
        if args.frames:
            collect_count = max(1, collect_count)

        episode_results = []
        model.eval()
        if args.num_actors > 1 and collect_count > 1:
            with ThreadPoolExecutor(max_workers=args.num_actors) as executor:
                futures = [
                    executor.submit(
                        run_episode, model, args, device, True)
                    for _ in range(collect_count)
                ]
                for future in as_completed(futures):
                    episode_results.append(future.result())
        else:
            for _ in range(collect_count):
                episode_results.append(
                    run_episode(model, args, device, train=True))
        model.train()

        for transitions, stats in episode_results:
            episode += 1
            buffer.extend(transitions)
            frames += len(transitions)
            recent.update(stats)
            recent['episodes'] += 1
            team0_reward = stats['team0_reward_x1000'] / 1000.0
            recent_team0_rewards.append(team0_reward)
            recent_abs_utilities.append(abs(team0_reward))

        if len(buffer) >= args.update_frames:
            last_update = dmc_update(model, optimizer, buffer, args, device)
            buffer.clear()

        should_log = episode <= 5 or episode % args.log_every == 0
        should_eval = next_eval is not None and frames >= next_eval
        eval_metrics = {}
        if should_eval:
            model.eval()
            eval_metrics.update(evaluate(model, args, device, 'eval'))
            write_eval_records(
                model, args, device, args.eval_record_jsonl,
                frames, episode)
            model.train()
            while frames >= next_eval:
                next_eval += args.eval_every

        if should_log or should_eval:
            elapsed = max(1e-6, time.time() - start_time)
            episodes_n = max(1, recent['episodes'])
            metrics = {
                'episode': episode,
                'frames': frames,
                'fps': frames / elapsed,
                'epsilon': args.epsilon,
                'buffer': len(buffer),
                'loss': last_update.get('loss'),
                'abs_error': last_update.get('abs_error'),
                'q_mean': last_update.get('q_mean'),
                'grad_norm': last_update.get('grad_norm'),
                'team0_reward_mean': (
                    sum(recent_team0_rewards)
                    / max(1, len(recent_team0_rewards))),
                'terminal_utility_abs_mean': (
                    sum(recent_abs_utilities)
                    / max(1, len(recent_abs_utilities))),
                'finish_rate': recent['finished'] / episodes_n,
                'full_hole_rate': recent['result_quan_dong'] / episodes_n,
                'half_hole_rate': recent['result_ban_dong'] / episodes_n,
                'stage_change_rate': recent['stage_changes'] / episodes_n,
                'team0_win_rate': recent['team0_wins'] / episodes_n,
                'team1_win_rate': recent['team1_wins'] / episodes_n,
                'team0_full_hole_rate': (
                    recent['team0_quan_dong'] / episodes_n),
                'team0_half_hole_rate': (
                    recent['team0_ban_dong'] / episodes_n),
                'team1_full_hole_rate': (
                    recent['team1_quan_dong'] / episodes_n),
                'team1_half_hole_rate': (
                    recent['team1_ban_dong'] / episodes_n),
                'invalid': recent['invalid'],
                'avg_candidates': (
                    recent['avg_candidates_x1000'] / max(1, episodes_n) / 1000.0),
                'avg_selected_q': (
                    recent['avg_selected_q_x1000'] / max(1, episodes_n) / 1000.0),
                'avg_greedy_q': (
                    recent['avg_greedy_q_x1000'] / max(1, episodes_n) / 1000.0),
                'plays': recent['plays'],
                'passes': recent['passes'],
                'cha_asks': recent['cha_asks'],
                'dian_asks': recent['dian_asks'],
                'cha_declines': recent['cha_declines'],
                'dian_declines': recent['dian_declines'],
                'explore_actions': recent['explore_actions'],
            }
            metrics['avg_q_gap'] = metrics['avg_greedy_q'] - metrics['avg_selected_q']
            action_total = max(1, metrics['plays'] + metrics['passes'])
            decision_total = max(
                1, metrics['plays'] + metrics['passes']
                + metrics['cha_asks'] + metrics['dian_asks'])
            metrics['pass_rate'] = metrics['passes'] / action_total
            metrics['explore_rate'] = metrics['explore_actions'] / decision_total
            metrics['cha_decline_rate'] = metrics['cha_declines'] / max(1, metrics['cha_asks'])
            metrics['dian_decline_rate'] = metrics['dian_declines'] / max(1, metrics['dian_asks'])
            metrics.update(eval_metrics)
            _log_line(metrics)
            _write_jsonl(args.log_jsonl, metrics)
            recent = Counter()

        while next_checkpoint is not None and frames >= next_checkpoint:
            path = _checkpoint_path(args.save, next_checkpoint)
            save_checkpoint(model, optimizer, path, frames, episode,
                            args, last_update)
            print('checkpoint已保存=%s' % path, flush=True)
            next_checkpoint += args.checkpoint_every

    if buffer:
        last_update = dmc_update(model, optimizer, buffer, args, device)
    save_checkpoint(model, optimizer, args.save, frames, episode,
                    args, last_update)
    print('最终checkpoint=%s 样本步数=%d 局数=%d' % (
        args.save, frames, episode), flush=True)


if __name__ == '__main__':
    main()
