"""Distributed actor-learner trainer for DouZero-4A4.

This script keeps one shared learner/replay buffer and runs many independent
actor processes. Actors refresh the latest teacher weights, collect complete
4A4 episodes, and push CPU transitions into a multiprocessing queue. The learner
trains one teacher model and one backend-compatible student model from the
shared replay buffer.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter, deque
from multiprocessing import Event, Process, Queue, set_start_method
from queue import Empty
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.room import LEVEL_ORDER
from rl.model import CardPolicyNetwork, torch
from scripts.train_douzero_4a4 import (
    Decision,
    dmc_update,
    evaluate,
    save_checkpoint,
    run_episode,
)


class DouZero4A4CardPolicyModel(torch.nn.Module):
    """Four position-specific teacher models using the same backend architecture."""

    def __init__(self):
        super().__init__()
        self.models = torch.nn.ModuleList(
            [CardPolicyNetwork() for _ in range(4)])

    def forward_for_seat(self, seat, state_vec, action_vecs, history_vecs):
        return self.models[seat % 4](state_vec, action_vecs, history_vecs)

    def forward(self, seat, state_vec, action_vecs, history_vecs):
        return self.forward_for_seat(seat, state_vec, action_vecs, history_vecs)


def _to_cpu_transition(item):
    return {
        'state_vec': item.state_vec.detach().cpu().tolist(),
        'history_vec': item.history_vec.detach().cpu().tolist(),
        'action_vecs': item.action_vecs.detach().cpu().tolist(),
        'action_index': item.action_index,
        'seat': item.seat,
        'reward': item.reward,
    }


def _from_packed_transition(item):
    return Decision(
        state_vec=torch.tensor(item['state_vec'], dtype=torch.float32),
        history_vec=torch.tensor(item['history_vec'], dtype=torch.float32),
        action_vecs=torch.tensor(item['action_vecs'], dtype=torch.float32),
        action_index=item['action_index'],
        seat=item['seat'],
        reward=item['reward'],
    )


def _actor_args(args):
    keys = [
        'level_rank', 'max_episode_steps', 'epsilon', 'epsilon_eval',
        'batch_size', 'max_grad_norm',
    ]
    data = {key: getattr(args, key) for key in keys}
    return SimpleNamespace(**data)


def _safe_load_teacher(model, path, device, last_mtime):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return last_mtime
    if last_mtime is not None and mtime <= last_mtime:
        return last_mtime
    try:
        payload = torch.load(path, map_location=device)
        state = payload.get('douzero_model_state_dict', payload)
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
    model = DouZero4A4CardPolicyModel().to(device)
    model.eval()
    last_mtime = None
    local_args = _actor_args(args)
    episodes = 0

    while not stop_event.is_set():
        last_mtime = _safe_load_teacher(
            model, weight_path, device, last_mtime)
        try:
            with torch.no_grad():
                transitions, stats = run_episode(
                    model, local_args, device, train=True)
            cpu_transitions = [_to_cpu_transition(t) for t in transitions]
            out_queue.put((actor_id, cpu_transitions, dict(stats)))
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


def _publish_teacher(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    torch.save({'douzero_model_state_dict': model.state_dict()}, tmp)
    os.replace(tmp, path)


def _write_jsonl(path, metrics):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + '\n')


def _fmt(value):
    if value is None:
        return 'NA'
    if isinstance(value, float):
        return '%.5f' % value
    return str(value)


def _log(metrics):
    labels = {
        'frames': '帧数',
        'episodes': '回合数',
        'fps': 'FPS',
        'replay': '重放队列长度',
        'queue': '交互队列长度',
        'queue_timeouts': '队列超时次数',
        'loss': '教师损失',
        'abs_error': '教师平均绝对误差',
        'backend_loss': '后端损失',
        'backend_abs_error': '后端平均绝对误差',
        'finish_rate': '完成率',
        'full_hole_rate': '全洞率',
        'half_hole_rate': '半洞率',
        'stage_change_rate': '阶段变化率',
        'invalid': '无效操作数',
        'actor_errors': '演员错误数',
        'avg_candidates': '平均候选动作数',
        'avg_selected_q': '平均选中Q值',
        'avg_greedy_q': '平均贪婪Q值',
        'team0_reward_mean': '队伍0平均奖励',
        'team1_reward_mean': '队伍1平均奖励',
        'transitions': '决策数',
        'plays': '出牌次数',
        'passes': 'Pass次数',
        'explore_actions': '探索动作数',
        'cha_asks': '查问次数',
        'cha_declines': '查拒绝次数',
        'dian_asks': '点问次数',
        'dian_declines': '点拒绝次数',
        'q_mean': '教师平均Q值',
        'grad_norm': '教师梯度范数',
        'backend_q_mean': '后端平均Q值',
        'backend_grad_norm': '后端梯度范数',
        'teacher_eval_model_team_win_rate': '教师评估队伍胜率',
        'backend_eval_model_team_win_rate': '后端评估队伍胜率',
    }
    keys = [
        'frames', 'episodes', 'fps', 'replay', 'queue', 'queue_timeouts',
        'loss', 'abs_error', 'q_mean', 'grad_norm',
        'backend_loss', 'backend_abs_error', 'backend_q_mean',
        'backend_grad_norm',
        'finish_rate', 'full_hole_rate', 'half_hole_rate',
        'stage_change_rate', 'invalid', 'actor_errors',
        'avg_candidates', 'avg_selected_q', 'avg_greedy_q',
        'team0_reward_mean', 'team1_reward_mean', 'transitions',
        'plays', 'passes', 'explore_actions',
        'cha_asks', 'cha_declines', 'dian_asks', 'dian_declines',
        'teacher_eval_model_team_win_rate',
        'backend_eval_model_team_win_rate',
    ]
    print(' '.join('%s=%s' % (labels.get(key, key), _fmt(metrics[key]))
                   for key in keys if key in metrics), flush=True)


def _checkpoint_path(base_path, frames):
    root, ext = os.path.splitext(base_path)
    return '%s_step_%d%s' % (root, frames, ext or '.pt')


def _save_split_teacher_checkpoints(path, teacher, args, frames,
                                   episodes, last_update):
    if not hasattr(teacher, 'models'):
        return
    root, ext = os.path.splitext(path)
    ext = ext or '.pt'
    for seat_idx, seat_model in enumerate(teacher.models):
        seat_path = '%s_seat%d%s' % (root, seat_idx, ext)
        os.makedirs(os.path.dirname(seat_path) or '.', exist_ok=True)
        torch.save({
            'model_state_dict': seat_model.state_dict(),
            'model_type': 'douzero_4a4_teacher_seat',
            'teacher_model_class': 'rl.model.CardPolicyNetwork',
            'global_step': frames,
            'episode': episodes,
            'args': vars(args),
            'metrics': last_update,
        }, seat_path)


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
        'avg_candidates': (
            recent['avg_candidates_x1000'] / episodes / 1000.0),
        'avg_selected_q': (
            recent['avg_selected_q_x1000'] / episodes / 1000.0),
        'avg_greedy_q': (
            recent['avg_greedy_q_x1000'] / episodes / 1000.0),
        'team0_reward_mean': (
            recent['team0_reward_x1000'] / episodes / 1000.0),
        'team1_reward_mean': (
            recent['team1_reward_x1000'] / episodes / 1000.0),
        'transitions': recent['transitions'],
        'plays': recent['plays'],
        'passes': recent['passes'],
        'explore_actions': recent['explore_actions'],
        'queue_timeouts': recent['queue_timeouts'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=100000000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--actor-devices', default='cuda:0,cuda:1,cuda:2')
    parser.add_argument('--actors-per-device', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260606)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backend-lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--replay-size', type=int, default=60000)
    parser.add_argument('--min-replay', type=int, default=8192)
    parser.add_argument('--update-frames', type=int, default=4096)
    parser.add_argument('--update-batches', type=int, default=8)
    parser.add_argument('--epsilon', type=float, default=0.08)
    parser.add_argument('--epsilon-eval', type=float, default=0.0)
    parser.add_argument('--max-grad-norm', type=float, default=40.0)
    parser.add_argument('--max-episode-steps', type=int, default=700)
    parser.add_argument('--level-rank', default='random',
                        choices=['random'] + LEVEL_ORDER)
    parser.add_argument('--save', default=os.path.join(
        ROOT, 'checkpoints', 'douzero_4a4_distributed_100x.pt'))
    parser.add_argument('--checkpoint-every', type=int, default=1000000)
    parser.add_argument('--eval-every', type=int, default=5000000)
    parser.add_argument('--eval-episodes', type=int, default=256)
    parser.add_argument('--log-every', type=int, default=100000)
    parser.add_argument('--log-jsonl', default=os.path.join(
        ROOT, 'logs', 'douzero_4a4_distributed_100x_metrics.jsonl'))
    parser.add_argument('--weight-publish-every', type=int, default=50000)
    parser.add_argument('--actor-refresh-episodes', type=int, default=20)
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

    teacher = DouZero4A4CardPolicyModel().to(learner_device)
    backend = CardPolicyNetwork().to(learner_device)
    teacher_optimizer = torch.optim.RMSprop(
        teacher.parameters(), lr=args.lr, momentum=0.0, eps=1e-5,
        alpha=0.99)
    backend_optimizer = torch.optim.RMSprop(
        backend.parameters(), lr=args.backend_lr, momentum=0.0, eps=1e-5,
        alpha=0.99)

    run_dir = os.path.join(ROOT, 'runtime', 'douzero_4a4_distributed')
    os.makedirs(run_dir, exist_ok=True)
    weight_path = os.path.join(run_dir, 'latest_teacher.pt')
    _publish_teacher(teacher, weight_path)

    replay = deque(maxlen=args.replay_size)
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

    frames = 0
    episodes = 0
    since_update = 0
    next_log = args.log_every if args.log_every > 0 else None
    next_eval = args.eval_every if args.eval_every > 0 else None
    next_checkpoint = (
        args.checkpoint_every if args.checkpoint_every > 0 else None)
    next_publish = (
        args.weight_publish_every if args.weight_publish_every > 0 else None)
    recent = Counter()
    last_update = {'updated': False}
    start = time.time()

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
            transitions = [_from_packed_transition(t) for t in transitions]
            replay.extend(transitions)
            frames += len(transitions)
            since_update += len(transitions)

            if len(replay) >= args.min_replay and since_update >= args.update_frames:
                sample_count = args.batch_size * args.update_batches
                batch = _sample_replay(replay, sample_count)
                teacher_update = dmc_update(
                    teacher, teacher_optimizer, batch, args, learner_device)
                backend_update = dmc_update(
                    backend, backend_optimizer, batch, args, learner_device)
                last_update = dict(teacher_update)
                for key, value in backend_update.items():
                    last_update['backend_' + key] = value
                since_update = 0

            if next_publish is not None and frames >= next_publish:
                _publish_teacher(teacher, weight_path)
                while frames >= next_publish:
                    next_publish += args.weight_publish_every

            should_log = next_log is not None and frames >= next_log
            should_eval = next_eval is not None and frames >= next_eval
            metrics = {}
            if should_eval:
                teacher.eval()
                backend.eval()
                metrics.update(evaluate(
                    teacher, args, learner_device, 'teacher_eval'))
                metrics.update(evaluate(
                    backend, args, learner_device, 'backend_eval'))
                teacher.train()
                backend.train()
                while frames >= next_eval:
                    next_eval += args.eval_every

            if should_log or should_eval:
                elapsed = max(1e-6, time.time() - start)
                metrics.update({
                    'frames': frames,
                    'episodes': episodes,
                    'fps': frames / elapsed,
                    'replay': len(replay),
                    'queue': queue.qsize() if hasattr(queue, 'qsize') else -1,
                    'loss': last_update.get('loss'),
                    'abs_error': last_update.get('abs_error'),
                    'backend_loss': last_update.get('backend_loss'),
                    'backend_abs_error': last_update.get('backend_abs_error'),
                    'invalid': recent['invalid'],
                    'actor_errors': recent['actor_errors'],
                    'cha_asks': recent['cha_asks'],
                    'cha_declines': recent['cha_declines'],
                    'dian_asks': recent['dian_asks'],
                    'dian_declines': recent['dian_declines'],
                })
                metrics.update(_aggregate_rates(recent))
                _log(metrics)
                _write_jsonl(args.log_jsonl, metrics)
                recent = Counter()
                while frames >= next_log:
                    next_log += args.log_every

            while next_checkpoint is not None and frames >= next_checkpoint:
                path = _checkpoint_path(args.save, next_checkpoint)
                save_checkpoint(
                    teacher, backend, teacher_optimizer, backend_optimizer,
                    path, frames, episodes, args, last_update)
                _save_split_teacher_checkpoints(
                    path, teacher, args, frames, episodes, last_update)
                print('已保存检查点=%s' % path, flush=True)
                next_checkpoint += args.checkpoint_every

        save_checkpoint(
            teacher, backend, teacher_optimizer, backend_optimizer,
            args.save, frames, episodes, args, last_update)
        _save_split_teacher_checkpoints(
            args.save, teacher, args, frames, episodes, last_update)
        print('最终检查点=%s 帧数=%d 回合数=%d' % (
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
