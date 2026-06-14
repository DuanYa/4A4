"""Distributed actor-learner trainer for DouZero-4A4.

This script keeps one shared learner and runs many independent actor processes.
Actors refresh the latest shared Transformer weights, collect complete 4A4
episodes, and push CPU transitions into a multiprocessing queue. The learner
updates on the most recent on-policy rollout and then clears that rollout,
keeping PPO closer to its intended data flow.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from multiprocessing import Event, Process, Queue, set_start_method
from queue import Empty
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.room import LEVEL_ORDER
from rl.model import ACTION_PAD_ID, HISTORY_DIM, CardPolicyNetwork, torch
from scripts.train_douzero_4a4 import (
    Decision,
    _load_resume,
    evaluate,
    ppo_update,
    save_checkpoint,
    run_episode,
    write_eval_records,
)


def _pack_cpu_transitions(transitions):
    if not transitions:
        return None
    max_actions = max(item.action_ids.shape[0] for item in transitions)
    actions = torch.full(
        (len(transitions), max_actions), ACTION_PAD_ID, dtype=torch.long)
    action_counts = []
    for row, item in enumerate(transitions):
        count = item.action_ids.shape[0]
        actions[row, :count] = item.action_ids.detach().cpu()
        action_counts.append(count)
    max_history = max([item.history_vec.shape[0] for item in transitions] + [1])
    history = torch.zeros(
        len(transitions), max_history, HISTORY_DIM, dtype=torch.float32)
    history[..., 4] = ACTION_PAD_ID
    history_counts = []
    for row, item in enumerate(transitions):
        count = item.history_vec.shape[0]
        if count > 0:
            history[row, :count] = item.history_vec.detach().cpu()
        history_counts.append(count)
    return {
        'state_vecs': torch.stack(
            [item.state_vec.detach().cpu() for item in transitions]).numpy(),
        'perfect_state_vecs': torch.stack(
            [item.perfect_state_vec.detach().cpu()
             for item in transitions]).numpy(),
        'history_vecs': history.numpy(),
        'history_counts': history_counts,
        'action_ids': actions.numpy(),
        'action_counts': action_counts,
        'action_indices': [item.action_index for item in transitions],
        'old_log_probs': [
            float(item.old_log_prob.detach().cpu().item())
            for item in transitions],
        'old_values': [
            float(item.old_value.detach().cpu().item())
            for item in transitions],
        'advantages': [item.advantage for item in transitions],
        'return_values': [item.return_value for item in transitions],
        'seats': [item.seat for item in transitions],
        'rewards': [item.reward for item in transitions],
    }


def _from_packed_transitions(items):
    if not items:
        return []
    state_vecs = torch.from_numpy(items['state_vecs']).float()
    perfect_state_vecs = torch.from_numpy(items['perfect_state_vecs']).float()
    history_vecs = torch.from_numpy(items['history_vecs']).float()
    action_ids = torch.from_numpy(items['action_ids']).long()
    decisions = []
    for row, count in enumerate(items['action_counts']):
        history_count = items.get('history_counts', [history_vecs.shape[1]])[row]
        decisions.append(Decision(
            state_vec=state_vecs[row],
            perfect_state_vec=perfect_state_vecs[row],
            history_vec=history_vecs[row, :history_count],
            action_ids=action_ids[row, :count],
            action_index=items['action_indices'][row],
            seat=items['seats'][row],
            old_log_prob=torch.tensor(items['old_log_probs'][row]),
            old_value=torch.tensor(items['old_values'][row]),
            advantage=items.get('advantages', [0.0])[row],
            return_value=items.get('return_values', items['rewards'])[row],
            reward=items['rewards'][row],
        ))
    return decisions


def _actor_args(args):
    keys = [
        'level_rank', 'max_episode_steps', 'epsilon', 'epsilon_eval',
        'batch_size', 'max_grad_norm',
        'ppo_clip', 'gae_gamma', 'gae_lambda', 'value_loss_coef',
        'entropy_coef', 'distance_reward_scale',
    ]
    data = {key: getattr(args, key) for key in keys}
    return SimpleNamespace(**data)


def _safe_load_model(model, path, device, last_mtime):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return last_mtime
    if last_mtime is not None and mtime <= last_mtime:
        return last_mtime
    try:
        payload = torch.load(path, map_location=device)
        state = payload.get('model_state_dict', payload)
        model.load_state_dict(state)
        model.eval()
        return mtime
    except Exception:
        return last_mtime


def actor_loop(actor_id, device_name, args_dict, weight_path, out_queue,
               stop_event):
    args = SimpleNamespace(**args_dict)
    if args.seed >= 0:
        seed = args.seed + actor_id * 9973
        random.seed(seed)
        torch.manual_seed(seed)

    if device_name.startswith('cuda') and not torch.cuda.is_available():
        device_name = 'cpu'
    device = torch.device(device_name)
    model = CardPolicyNetwork().to(device)
    model.eval()
    last_mtime = None
    local_args = _actor_args(args)
    episodes = 0

    while not stop_event.is_set():
        last_mtime = _safe_load_model(
            model, weight_path, device, last_mtime)
        try:
            model_team = None
            if random.random() < float(getattr(args, 'rule_opponent_rate', 0.0)):
                model_team = random.randrange(2)
            with torch.no_grad():
                transitions, stats = run_episode(
                    model, local_args, device, train=True,
                    model_team=model_team)
            if model_team is None:
                stats['self_play_episodes'] += 1
            else:
                stats['rule_opponent_episodes'] += 1
                stats['train_model_team_%d' % model_team] += 1
            out_queue.put((
                actor_id, _pack_cpu_transitions(transitions), dict(stats)))
            episodes += 1
            if episodes % max(1, args.actor_refresh_episodes) == 0:
                last_mtime = None
        except Exception as exc:
            try:
                out_queue.put((actor_id, [], {
                    'actor_errors': 1,
                    'actor_error_message': repr(exc),
                }), timeout=30)
            except Exception:
                pass
            time.sleep(1.0)


def _publish_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    torch.save({'model_state_dict': model.state_dict()}, tmp)
    os.replace(tmp, path)


def _write_jsonl(path, metrics):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    cn_metrics = {CN_KEYS.get(key, key): value for key, value in metrics.items()}
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(cn_metrics, ensure_ascii=False, sort_keys=True) + '\n')


def _fmt(value):
    if value is None:
        return 'NA'
    if isinstance(value, float):
        return '%.5f' % value
    return str(value)


CN_KEYS = {
    'frames': '样本步数',
    'episodes': '局数',
    'fps': '每秒样本',
    'replay': '旧回放池样本',
    'rollout': 'on-policy样本',
    'queue': '采样队列',
    'loss': '训练损失',
    'policy_loss': '策略损失',
    'value_loss': '价值损失',
    'entropy': '策略熵',
    'approx_kl': '近似KL',
    'clip_frac': '裁剪比例',
    'adv_mean': '优势均值',
    'adv_std': '优势标准差',
    'abs_error': '平均绝对误差',
    'q_mean': '平均Q值',
    'grad_norm': '梯度范数',
    'update_ms': '更新耗时毫秒',
    'update_batches_done': '更新批次数',
    'update_samples': '更新样本数',
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
    'actor_errors': '采样进程错误数',
    'queue_timeouts': '采样队列超时数',
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
    'cha_declines': '不叉次数',
    'cha_decline_rate': '不叉率',
    'dian_asks': '点牌询问数',
    'dian_declines': '不点次数',
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
    'self_play_rate': '自博弈训练占比',
    'rule_opponent_rate_seen': '规则对手训练占比',
}


def _log(metrics):
    keys = [
        'frames', 'episodes', 'fps', 'rollout', 'replay', 'queue',
        'loss', 'policy_loss', 'value_loss', 'entropy',
        'approx_kl', 'clip_frac', 'adv_mean', 'adv_std',
        'abs_error', 'q_mean',
        'grad_norm',
        'update_ms', 'update_batches_done', 'update_samples',
        'finish_rate', 'full_hole_rate', 'half_hole_rate',
        'stage_change_rate',
        'team0_win_rate', 'team1_win_rate',
        'team0_full_hole_rate', 'team0_half_hole_rate',
        'team1_full_hole_rate', 'team1_half_hole_rate',
        'self_play_rate', 'rule_opponent_rate_seen',
        'invalid', 'actor_errors', 'queue_timeouts',
        'avg_candidates', 'avg_selected_q', 'avg_greedy_q', 'avg_q_gap',
        'plays', 'passes', 'pass_rate', 'explore_actions', 'explore_rate',
        'cha_asks', 'cha_declines', 'cha_decline_rate',
        'dian_asks', 'dian_declines', 'dian_decline_rate',
        'eval_model_team_win_rate', 'eval_model_team_full_hole_rate',
        'eval_model_team_half_hole_rate', 'eval_rule_team_win_rate',
        'eval_rule_team_full_hole_rate', 'eval_rule_team_half_hole_rate',
        'eval_full_hole_rate', 'eval_half_hole_rate',
        'eval_truncated_rate', 'eval_invalid',
    ]
    print(' '.join('%s=%s' % (CN_KEYS.get(key, key), _fmt(metrics[key]))
                   for key in keys if key in metrics), flush=True)


def _checkpoint_path(base_path, frames):
    root, ext = os.path.splitext(base_path)
    return '%s_step_%d%s' % (root, frames, ext or '.pt')


def _sample_replay(replay, count):
    if len(replay) <= count:
        return list(replay)
    return random.sample(list(replay), count)


def _aggregate_rates(recent):
    episodes = max(1, recent['episodes'])
    return {
        'finish_rate': recent['finished'] / episodes,
        'full_hole_rate': recent['result_quan_dong'] / episodes,
        'half_hole_rate': recent['result_ban_dong'] / episodes,
        'stage_change_rate': recent['stage_changes'] / episodes,
        'team0_win_rate': recent['team0_wins'] / episodes,
        'team1_win_rate': recent['team1_wins'] / episodes,
        'team0_full_hole_rate': recent['team0_quan_dong'] / episodes,
        'team0_half_hole_rate': recent['team0_ban_dong'] / episodes,
        'team1_full_hole_rate': recent['team1_quan_dong'] / episodes,
        'team1_half_hole_rate': recent['team1_ban_dong'] / episodes,
        'self_play_rate': recent['self_play_episodes'] / episodes,
        'rule_opponent_rate_seen': (
            recent['rule_opponent_episodes'] / episodes),
        'avg_candidates': (
            recent['avg_candidates_x1000'] / episodes / 1000.0),
        'avg_selected_q': (
            recent['avg_selected_q_x1000'] / episodes / 1000.0),
        'avg_greedy_q': (
            recent['avg_greedy_q_x1000'] / episodes / 1000.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=100000000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--actor-devices', default='cuda:0,cuda:1,cuda:2')
    parser.add_argument('--actors-per-device', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260606)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--replay-size', type=int, default=60000)
    parser.add_argument('--min-replay', type=int, default=8192)
    parser.add_argument('--update-frames', type=int, default=4096)
    parser.add_argument('--update-batches', type=int, default=8)
    parser.add_argument('--epsilon', type=float, default=0.08)
    parser.add_argument('--epsilon-eval', type=float, default=0.0)
    parser.add_argument('--max-grad-norm', type=float, default=40.0)
    parser.add_argument('--ppo-clip', type=float, default=0.2)
    parser.add_argument('--ppo-epochs', type=int, default=2)
    parser.add_argument('--gae-gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--value-loss-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--distance-reward-scale', type=float, default=0.1)
    parser.add_argument('--max-episode-steps', type=int, default=700)
    parser.add_argument('--level-rank', default='random',
                        choices=['random'] + LEVEL_ORDER)
    parser.add_argument('--save', default=os.path.join(
        ROOT, 'checkpoints', 'douzero_4a4_distributed_100x.pt'))
    parser.add_argument('--resume', default='')
    parser.add_argument('--checkpoint-every', type=int, default=1000000)
    parser.add_argument('--eval-every', type=int, default=5000000)
    parser.add_argument('--eval-episodes', type=int, default=256)
    parser.add_argument('--eval-record-episodes', type=int, default=4)
    parser.add_argument('--eval-record-jsonl', default=os.path.join(
        ROOT, 'logs', 'douzero_4a4_distributed_eval_records.jsonl'))
    parser.add_argument('--log-every', type=int, default=100000)
    parser.add_argument('--log-jsonl', default=os.path.join(
        ROOT, 'logs', 'douzero_4a4_distributed_100x_metrics.jsonl'))
    parser.add_argument('--weight-publish-every', type=int, default=50000)
    parser.add_argument('--actor-refresh-episodes', type=int, default=20)
    parser.add_argument('--rule-opponent-rate', type=float, default=0.5)
    parser.add_argument('--queue-size', type=int, default=64)
    parser.add_argument('--torch-threads', type=int, default=1)
    args = parser.parse_args()

    if torch is None:
        raise RuntimeError('distributed training requires torch')
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.seed >= 0:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        args.device = 'cpu'
    learner_device = torch.device(args.device)

    model = CardPolicyNetwork().to(learner_device)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.lr, momentum=0.0, eps=1e-5,
        alpha=0.99)
    frames, episodes = _load_resume(
        model, optimizer, args.resume, learner_device)
    if args.resume and frames > 0:
        print('已从checkpoint恢复=%s 样本步数=%d 局数=%d' % (
            args.resume, frames, episodes), flush=True)

    run_dir = os.path.join(ROOT, 'runtime', 'douzero_4a4_distributed')
    os.makedirs(run_dir, exist_ok=True)
    weight_path = os.path.join(run_dir, 'latest_model.pt')
    _publish_model(model, weight_path)

    rollout = []
    queue = Queue(maxsize=args.queue_size)
    stop_event = Event()
    actor_devices = [item.strip() for item in args.actor_devices.split(',')
                     if item.strip()]
    actors = []
    actor_args = vars(args)
    actor_id = 0
    for device_name in actor_devices:
        for _ in range(args.actors_per_device):
            proc = Process(
                target=actor_loop,
                args=(actor_id, device_name, actor_args, weight_path,
                      queue, stop_event),
                daemon=True)
            proc.start()
            actors.append(proc)
            actor_id += 1

    since_update = 0
    next_log = (
        ((frames // args.log_every) + 1) * args.log_every
        if args.log_every > 0 else None)
    next_eval = (
        ((frames // args.eval_every) + 1) * args.eval_every
        if args.eval_every > 0 else None)
    next_checkpoint = (
        ((frames // args.checkpoint_every) + 1) * args.checkpoint_every
        if args.checkpoint_every > 0 else None)
    next_publish = (
        ((frames // args.weight_publish_every) + 1)
        * args.weight_publish_every
        if args.weight_publish_every > 0 else None)
    recent = Counter()
    last_update = {'updated': False}
    start = time.time()
    start_frames = frames

    try:
        while frames < args.frames:
            try:
                _, transitions, stats = queue.get(timeout=30)
            except Empty:
                recent['queue_timeouts'] += 1
                continue

            stats = Counter(stats)
            recent.update(stats)
            recent['episodes'] += 1
            episodes += 1
            transitions = _from_packed_transitions(transitions)
            rollout.extend(transitions)
            frames += len(transitions)
            since_update += len(transitions)

            min_rollout = max(
                1, args.update_frames,
                args.min_replay if args.min_replay > 0 else 0)
            if len(rollout) >= min_rollout and since_update >= args.update_frames:
                batch = list(rollout)
                rollout.clear()
                update_start = time.time()
                last_update = ppo_update(
                    model, optimizer, batch, args, learner_device)
                last_update['update_ms'] = (
                    time.time() - update_start) * 1000.0
                last_update['update_samples'] = (
                    len(batch) * max(1, int(getattr(args, 'ppo_epochs', 1))))
                since_update = 0
                _publish_model(model, weight_path)

            if next_publish is not None and frames >= next_publish:
                _publish_model(model, weight_path)
                while frames >= next_publish:
                    next_publish += args.weight_publish_every

            should_log = next_log is not None and frames >= next_log
            should_eval = next_eval is not None and frames >= next_eval
            metrics = {}
            if should_eval:
                model.eval()
                metrics.update(evaluate(
                    model, args, learner_device, 'eval'))
                write_eval_records(
                    model, args, learner_device, args.eval_record_jsonl,
                    frames, episodes)
                model.train()
                while frames >= next_eval:
                    next_eval += args.eval_every

            if should_log or should_eval:
                elapsed = max(1e-6, time.time() - start)
                metrics.update({
                    'frames': frames,
                    'episodes': episodes,
                    'fps': max(0, frames - start_frames) / elapsed,
                    'rollout': len(rollout),
                    'replay': 0,
                    'queue': queue.qsize() if hasattr(queue, 'qsize') else -1,
                    'loss': last_update.get('loss'),
                    'policy_loss': last_update.get('policy_loss'),
                    'value_loss': last_update.get('value_loss'),
                    'entropy': last_update.get('entropy'),
                    'approx_kl': last_update.get('approx_kl'),
                    'clip_frac': last_update.get('clip_frac'),
                    'adv_mean': last_update.get('adv_mean'),
                    'adv_std': last_update.get('adv_std'),
                    'abs_error': last_update.get('abs_error'),
                    'q_mean': last_update.get('q_mean'),
                    'grad_norm': last_update.get('grad_norm'),
                    'update_ms': last_update.get('update_ms'),
                    'update_batches_done': last_update.get(
                        'update_batches_done'),
                    'update_samples': last_update.get('update_samples'),
                    'invalid': recent['invalid'],
                    'actor_errors': recent['actor_errors'],
                    'queue_timeouts': recent['queue_timeouts'],
                    'plays': recent['plays'],
                    'passes': recent['passes'],
                    'explore_actions': recent['explore_actions'],
                    'cha_asks': recent['cha_asks'],
                    'cha_declines': recent['cha_declines'],
                    'dian_asks': recent['dian_asks'],
                    'dian_declines': recent['dian_declines'],
                })
                metrics.update(_aggregate_rates(recent))
                metrics['avg_q_gap'] = (
                    metrics.get('avg_greedy_q', 0.0)
                    - metrics.get('avg_selected_q', 0.0))
                action_total = max(1, metrics['plays'] + metrics['passes'])
                decision_total = max(
                    1, metrics['plays'] + metrics['passes']
                    + metrics['cha_asks'] + metrics['dian_asks'])
                metrics['pass_rate'] = metrics['passes'] / action_total
                metrics['explore_rate'] = (
                    metrics['explore_actions'] / decision_total)
                metrics['cha_decline_rate'] = (
                    metrics['cha_declines'] / max(1, metrics['cha_asks']))
                metrics['dian_decline_rate'] = (
                    metrics['dian_declines'] / max(1, metrics['dian_asks']))
                _log(metrics)
                _write_jsonl(args.log_jsonl, metrics)
                recent = Counter()
                while frames >= next_log:
                    next_log += args.log_every

            while next_checkpoint is not None and frames >= next_checkpoint:
                path = _checkpoint_path(args.save, next_checkpoint)
                save_checkpoint(
                    model, optimizer, path, frames, episodes, args, last_update)
                print('checkpoint已保存=%s' % path, flush=True)
                next_checkpoint += args.checkpoint_every

        save_checkpoint(
            model, optimizer, args.save, frames, episodes, args, last_update)
        print('最终checkpoint=%s 样本步数=%d 局数=%d' % (
            args.save, frames, episodes), flush=True)
    finally:
        stop_event.set()
        for proc in actors:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


if __name__ == '__main__':
    set_start_method('spawn', force=True)
    main()
