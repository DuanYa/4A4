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

from douzero_4a4.model import DouZero4A4Model
from douzero_4a4.rewards import seat_rewards
from models.game import Game, GamePhase
from models.hand_type import HandType, HandCategory, identify_hand
from models.player import Player
from models.room import LEVEL_ORDER
from rl.actions import enumerate_legal_actions
from rl.features import encode_action, encode_history, encode_state
from rl.model import CardPolicyNetwork, torch
from models import ai_search


@dataclass
class Decision:
    state_vec: object
    history_vec: object
    action_vecs: object
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


def _select_model_action(model, state, hand, level_rank, seat, last_ht,
                         is_free_play, epsilon, device, train=True):
    actions = enumerate_legal_actions(
        hand, level_rank, last_ht, is_free_play)
    if not actions:
        return {'type': 'pass', 'indices': [], 'hand_type': None}, None, {}

    state_vec = torch.tensor(
        encode_state(state, hand, level_rank, seat),
        dtype=torch.float32, device=device)
    history_vec = torch.tensor(
        encode_history(state, seat),
        dtype=torch.float32, device=device)
    action_vecs = torch.tensor(
        [encode_action(action, hand, level_rank) for action in actions],
        dtype=torch.float32, device=device)

    with torch.no_grad():
        q_values = _forward_action_values(
            model, seat, state_vec, action_vecs, history_vec)
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
            action_vecs=action_vecs.detach(),
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


def _random_level(configured):
    if configured and configured != 'random':
        return configured
    return random.choice(LEVEL_ORDER)


def run_episode(model, args, device, train=True, model_team=None):
    level_rank = _random_level(args.level_rank)
    on_stage_team = random.randrange(2)
    players = [Player('dz4a4-%d' % i, 'DZ4A4-%d' % i, i)
               for i in range(4)]
    game = Game(players, level_rank, on_stage_team)
    game.start()

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
            do_cha = _training_should_cha(game, game.cha_asking_seat)
            if not do_cha:
                stats['cha_declines'] += 1
            game.respond_cha(game.cha_asking_seat, do_cha)
            continue

        if game.phase == GamePhase.DIAN_ASKING:
            stats['dian_asks'] += 1
            do_dian = _training_should_dian(game, game.dian_asking_seat)
            if not do_dian:
                stats['dian_declines'] += 1
            game.respond_dian(game.dian_asking_seat, do_dian)
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
    return transitions, stats


def dmc_update(model, optimizer, transitions, args, device):
    if not transitions:
        return {'updated': False}
    random.shuffle(transitions)
    losses = []
    abs_errors = []
    q_means = []
    grad_norm = 0.0

    for start in range(0, len(transitions), args.batch_size):
        batch = transitions[start:start + args.batch_size]
        preds = []
        targets = []
        for item in batch:
            item_state = item.state_vec.to(device)
            item_actions = item.action_vecs.to(device)
            item_history = item.history_vec.to(device)
            q_values = _forward_action_values(
                model, item.seat, item_state, item_actions,
                item_history)
            preds.append(q_values[item.action_index].reshape(()))
            targets.append(torch.tensor(item.reward, dtype=torch.float32,
                                        device=device))
        pred = torch.stack(preds)
        target = torch.stack(targets)
        loss = torch.nn.functional.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm)
        optimizer.step()

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
    }


def save_checkpoint(douzero_model, backend_model, douzero_optimizer,
                    backend_optimizer, path, frames, episodes, args,
                    metrics):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        # Existing backend loads exactly this key into rl.model.CardPolicyNetwork.
        'model_state_dict': backend_model.state_dict(),
        'optimizer_state_dict': backend_optimizer.state_dict(),
        'douzero_model_state_dict': douzero_model.state_dict(),
        'douzero_optimizer_state_dict': douzero_optimizer.state_dict(),
        'global_step': frames,
        'episode': episodes,
        'model_type': 'douzero_4a4_dmc_backend_compatible',
        'teacher_model_class': 'douzero_4a4.model.DouZero4A4Model',
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


def _load_resume(douzero_model, backend_model, douzero_optimizer,
                 backend_optimizer, path, device):
    if not path or not os.path.exists(path):
        return 0, 0
    payload = torch.load(path, map_location=device)
    if 'douzero_model_state_dict' in payload:
        douzero_model.load_state_dict(payload['douzero_model_state_dict'])
    elif 'model_state_dict' in payload:
        _load_compatible(douzero_model, payload['model_state_dict'])
    if 'model_state_dict' in payload:
        _load_compatible(backend_model, payload['model_state_dict'])
    if 'douzero_optimizer_state_dict' in payload:
        douzero_optimizer.load_state_dict(
            payload['douzero_optimizer_state_dict'])
    if 'optimizer_state_dict' in payload:
        backend_optimizer.load_state_dict(payload['optimizer_state_dict'])
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
    return {
        metric_prefix + '_episodes': total['episodes'],
        metric_prefix + '_model_team_win_rate': _counter_rate(
            total, 'model_team_wins'),
        metric_prefix + '_full_hole_rate': _counter_rate(
            total, 'result_quan_dong'),
        metric_prefix + '_half_hole_rate': _counter_rate(
            total, 'result_ban_dong'),
        metric_prefix + '_truncated_rate': _counter_rate(
            total, 'result_truncated'),
        metric_prefix + '_invalid': total['invalid'],
    }


def _format_float(value):
    if value is None:
        return 'NA'
    return '%.5f' % value


def _log_line(metrics):
    keys = [
        'episode', 'frames', 'fps', 'epsilon', 'buffer',
        'loss', 'abs_error', 'q_mean', 'grad_norm',
        'backend_loss', 'backend_abs_error', 'backend_q_mean',
        'backend_grad_norm',
        'team0_reward_mean', 'terminal_utility_abs_mean', 'finish_rate', 'full_hole_rate',
        'half_hole_rate', 'stage_change_rate', 'invalid',
        'avg_candidates', 'avg_selected_q', 'avg_greedy_q',
        'plays', 'passes', 'cha_asks', 'dian_asks',
        'cha_declines', 'dian_declines',
        'teacher_eval_model_team_win_rate',
        'backend_eval_model_team_win_rate',
    ]
    parts = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            value = _format_float(value)
        parts.append('%s=%s' % (key, value))
    print(' '.join(parts), flush=True)


def _write_jsonl(path, metrics):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=200000)
    parser.add_argument('--episodes', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backend-lr', type=float, default=1e-4)
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
    args = parser.parse_args()

    if torch is None:
        raise RuntimeError('训练需要 torch，请在远端执行: conda activate torch')
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)

    douzero_model = DouZero4A4Model().to(device)
    backend_model = CardPolicyNetwork().to(device)
    douzero_optimizer = torch.optim.RMSprop(
        douzero_model.parameters(), lr=args.lr, momentum=0.0,
        eps=1e-5, alpha=0.99)
    backend_optimizer = torch.optim.RMSprop(
        backend_model.parameters(), lr=args.backend_lr, momentum=0.0,
        eps=1e-5, alpha=0.99)
    frames, episode = _load_resume(
        douzero_model, backend_model, douzero_optimizer,
        backend_optimizer, args.resume, device)
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
        douzero_model.eval()
        if args.num_actors > 1 and collect_count > 1:
            with ThreadPoolExecutor(max_workers=args.num_actors) as executor:
                futures = [
                    executor.submit(
                        run_episode, douzero_model, args, device, True)
                    for _ in range(collect_count)
                ]
                for future in as_completed(futures):
                    episode_results.append(future.result())
        else:
            for _ in range(collect_count):
                episode_results.append(
                    run_episode(douzero_model, args, device, train=True))
        douzero_model.train()

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
            teacher_update = dmc_update(
                douzero_model, douzero_optimizer, buffer, args, device)
            backend_update = dmc_update(
                backend_model, backend_optimizer, buffer, args, device)
            last_update = dict(teacher_update)
            for key, value in backend_update.items():
                last_update['backend_' + key] = value
            buffer.clear()

        should_log = episode <= 5 or episode % args.log_every == 0
        should_eval = next_eval is not None and frames >= next_eval
        eval_metrics = {}
        if should_eval:
            douzero_model.eval()
            backend_model.eval()
            eval_metrics.update(evaluate(
                douzero_model, args, device, 'teacher_eval'))
            eval_metrics.update(evaluate(
                backend_model, args, device, 'backend_eval'))
            douzero_model.train()
            backend_model.train()
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
                'backend_loss': last_update.get('backend_loss'),
                'backend_abs_error': last_update.get('backend_abs_error'),
                'backend_q_mean': last_update.get('backend_q_mean'),
                'backend_grad_norm': last_update.get('backend_grad_norm'),
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
            metrics.update(eval_metrics)
            _log_line(metrics)
            _write_jsonl(args.log_jsonl, metrics)
            recent = Counter()

        while next_checkpoint is not None and frames >= next_checkpoint:
            path = _checkpoint_path(args.save, next_checkpoint)
            save_checkpoint(douzero_model, backend_model, douzero_optimizer,
                            backend_optimizer, path, frames, episode,
                            args, last_update)
            print('checkpoint_saved=%s' % path, flush=True)
            next_checkpoint += args.checkpoint_every

    if buffer:
        teacher_update = dmc_update(
            douzero_model, douzero_optimizer, buffer, args, device)
        backend_update = dmc_update(
            backend_model, backend_optimizer, buffer, args, device)
        last_update = dict(teacher_update)
        for key, value in backend_update.items():
            last_update['backend_' + key] = value
    save_checkpoint(douzero_model, backend_model, douzero_optimizer,
                    backend_optimizer, args.save, frames, episode,
                    args, last_update)
    print('final_checkpoint=%s frames=%d episode=%d' % (
        args.save, frames, episode), flush=True)


if __name__ == '__main__':
    main()
