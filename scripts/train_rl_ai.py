"""
四幺四 PPO 强化学习训练脚本。

针对旧版REINFORCE收敛差的问题，本脚本改为 PPO Actor-Critic：
- Actor：在当前合法候选动作中输出动作概率；
- Critic：估计当前局面价值，降低策略梯度方差；
- PPO裁剪目标：避免单次更新过大导致策略崩坏；
- 历史Transformer特征：输入最近出牌历史，帮助推断各家剩余牌；
- 每 save_every 个环境决策step保存一次checkpoint。

运行示例：
    python scripts/train_rl_ai.py --steps 200000 --save checkpoints/default_rl.pt
    python scripts/train_rl_ai.py --episodes 500 --save checkpoints/default_rl.pt
"""
import argparse
import os
import sys
from dataclasses import dataclass
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.game import Game, GamePhase
from models.player import Player
from models.hand_type import HandType, HandCategory
from rl.actions import enumerate_legal_actions
from rl.features import encode_state, encode_action, encode_history
from rl.model import HISTORY_DIM, CardPolicyNetwork, torch


@dataclass
class Transition:
    """PPO训练所需的单步轨迹数据"""
    state_vec: object
    history_vec: object
    action_vecs: object
    action_index: int
    old_log_prob: object
    old_value: object
    reward: float = 0.0


def _dict_to_hand_type(data):
    """训练中从状态字典恢复HandType"""
    if not data:
        return None
    return HandType(
        HandCategory[data['category']],
        [],
        data.get('key_value', 0),
        data.get('length', 0),
    )


def _sample_action(model, state, hand, level_rank, seat,
                   last_ht, is_free_play):
    """从合法动作中按当前策略采样，并返回PPO轨迹"""
    actions = enumerate_legal_actions(
        hand, level_rank, last_ht, is_free_play)
    if not actions:
        return {'type': 'pass', 'indices': [], 'hand_type': None}, None

    state_vec = torch.tensor(
        encode_state(state, hand, level_rank, seat),
        dtype=torch.float32)
    history_rows = encode_history(state, seat)
    if history_rows:
        history_vec = torch.tensor(history_rows, dtype=torch.float32)
    else:
        history_vec = torch.empty((0, HISTORY_DIM), dtype=torch.float32)
    action_vecs = torch.tensor(
        [encode_action(a, hand, level_rank) for a in actions],
        dtype=torch.long)

    with torch.no_grad():
        logits, value = model.forward_actor_critic(
            state_vec, action_vecs, history_vec)
        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample()
        old_log_prob = dist.log_prob(idx)

    trans = Transition(
        state_vec=state_vec,
        history_vec=history_vec,
        action_vecs=action_vecs,
        action_index=int(idx.item()),
        old_log_prob=old_log_prob.detach(),
        old_value=value.detach(),
    )
    return actions[trans.action_index], trans


def _seat_relative_score(order_map, seat):
    """计算某座位相对两个对手的胜负分"""
    own_order = order_map[seat]
    opp_orders = [order_map[(seat + 1) % 4],
                  order_map[(seat + 3) % 4]]
    beaten = sum(1 for opp_order in opp_orders
                 if own_order < opp_order)
    if beaten == 2:
        return 1.0
    if beaten == 1:
        return 0.0
    return -1.0


def _finish_rewards(game):
    """
    区分自己和队友的终局奖励。

    自己赢但队友输只算平局，训练目标是：
    先保证自己有竞争力，再在此前提下帮助队友获得更好名次。
    """
    rewards = [0.0, 0.0, 0.0, 0.0]
    order = game.finish_order
    if len(order) < 4:
        return rewards

    order_map = {seat: rank for rank, seat in enumerate(order)}
    for seat in range(4):
        mate = (seat + 2) % 4
        own_score = _seat_relative_score(order_map, seat)
        mate_score = _seat_relative_score(order_map, mate)

        if own_score > 0 and mate_score > 0:
            rewards[seat] = 1.5
        elif own_score > 0 and mate_score == 0:
            rewards[seat] = 1.0
        elif own_score > 0 and mate_score < 0:
            rewards[seat] = 0.0
        elif own_score == 0 and mate_score > 0:
            rewards[seat] = 0.4
        elif own_score == 0 and mate_score == 0:
            rewards[seat] = 0.0
        elif own_score < 0 and mate_score > 0:
            rewards[seat] = -0.4
        else:
            rewards[seat] = -1.2

        if not (own_score > 0 and mate_score < 0):
            rewards[seat] += (3 - order_map[seat]) * 0.05
    return rewards


def run_episode(model, level_rank='3'):
    """执行一局自博弈，返回带终局奖励的轨迹和调试统计"""
    players = [Player('rl-%d' % i, 'RL-%d' % i, i)
               for i in range(4)]
    game = Game(players, level_rank, 0)
    game.start()
    trajectories = [[] for _ in range(4)]
    step_limit = 600
    decisions = 0
    stats = {
        'plays': 0,
        'passes': 0,
        'cha': 0,
        'dian': 0,
        'invalid': 0,
        'action_candidates': [],
        'hand_types': Counter(),
    }

    for _ in range(step_limit):
        if game.phase == GamePhase.ROUND_END:
            break

        if game.phase == GamePhase.CHA_ASKING:
            stats['cha'] += 1
            game.respond_cha(game.cha_asking_seat, True)
            continue

        if game.phase == GamePhase.DIAN_ASKING:
            stats['dian'] += 1
            game.respond_dian(game.dian_asking_seat, True)
            continue

        if game.phase != GamePhase.PLAYING:
            continue

        seat = game.current_player_seat
        player = players[seat]
        state = game.get_state(for_seat=seat)
        last_ht = _dict_to_hand_type(state.get('last_hand_type'))
        action, trans = _sample_action(
            model, state, player.hand, level_rank, seat,
            last_ht, state.get('is_free_play', False))
        if trans is not None:
            trajectories[seat].append(trans)
            decisions += 1
            stats['action_candidates'].append(
                int(trans.action_vecs.shape[0]))

        if action['type'] == 'pass':
            stats['passes'] += 1
            result = game.player_pass(seat)
        else:
            stats['plays'] += 1
            if action.get('hand_type'):
                stats['hand_types'][action['hand_type'].category.name] += 1
            result = game.play_cards(seat, action['indices'])

        if not result.get('success'):
            stats['invalid'] += 1
            if not state.get('is_free_play'):
                game.player_pass(seat)

    rewards = _finish_rewards(game)
    transitions = []
    for seat, logs in enumerate(trajectories):
        for trans in logs:
            trans.reward = rewards[seat]
            transitions.append(trans)

    avg_candidates = 0.0
    if stats['action_candidates']:
        avg_candidates = sum(stats['action_candidates']) / len(stats['action_candidates'])
    stats['avg_candidates'] = avg_candidates
    stats['finished'] = game.phase == GamePhase.ROUND_END
    stats['finish_order'] = list(game.finish_order)
    stats['transition_count'] = len(transitions)
    return transitions, rewards, decisions, stats


def _empty_update_metrics(updated=False):
    """构造统一的PPO日志指标，避免未更新时误读loss=0"""
    return {
        'updated': updated,
        'loss': None,
        'policy_loss': None,
        'value_loss': None,
        'entropy': None,
        'approx_kl': None,
        'clip_frac': None,
        'grad_norm': None,
        'adv_mean': None,
        'adv_std': None,
        'reward_mean': None,
    }


def ppo_update(model, optimizer, transitions, clip_eps=0.2,
               value_coef=0.5, entropy_coef=0.01, epochs=4):
    """使用PPO裁剪目标更新模型，并返回详细训练指标"""
    if not transitions:
        return _empty_update_metrics(False)

    total = Counter()
    update_count = 0
    grad_norm_value = 0.0
    rewards_all = torch.tensor(
        [t.reward for t in transitions], dtype=torch.float32)
    old_values_all = torch.stack([
        t.old_value.float().reshape(()) for t in transitions])
    advantages_all = rewards_all - old_values_all

    for _ in range(epochs):
        losses = []
        policy_losses = []
        value_losses = []
        entropies = []
        kls = []
        clip_flags = []
        for trans in transitions:
            logits, value = model.forward_actor_critic(
                trans.state_vec, trans.action_vecs, trans.history_vec)
            dist = torch.distributions.Categorical(logits=logits)
            action_tensor = torch.tensor(trans.action_index)
            log_prob = dist.log_prob(action_tensor)
            entropy = dist.entropy()
            reward = torch.tensor(trans.reward, dtype=torch.float32)
            advantage = reward - trans.old_value

            ratio = torch.exp(log_prob - trans.old_log_prob)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            clipped = clipped_ratio * advantage
            policy_loss = -torch.min(ratio * advantage, clipped)
            value_loss = (value - reward).pow(2)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            losses.append(loss)
            policy_losses.append(policy_loss.detach())
            value_losses.append(value_loss.detach())
            entropies.append(entropy.detach())
            kls.append((trans.old_log_prob - log_prob).detach())
            clip_flags.append((torch.abs(ratio - 1.0) > clip_eps).float().detach())

        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)
        optimizer.step()

        update_count += 1
        grad_norm_value = float(grad_norm)
        total['loss'] += float(loss.item())
        total['policy_loss'] += float(torch.stack(policy_losses).mean().item())
        total['value_loss'] += float(torch.stack(value_losses).mean().item())
        total['entropy'] += float(torch.stack(entropies).mean().item())
        total['approx_kl'] += float(torch.stack(kls).mean().item())
        total['clip_frac'] += float(torch.stack(clip_flags).mean().item())

    metrics = _empty_update_metrics(True)
    for key in ('loss', 'policy_loss', 'value_loss',
                'entropy', 'approx_kl', 'clip_frac'):
        metrics[key] = total[key] / max(1, update_count)
    metrics['grad_norm'] = grad_norm_value
    metrics['adv_mean'] = float(advantages_all.mean().item())
    metrics['adv_std'] = float(advantages_all.std(unbiased=False).item())
    metrics['reward_mean'] = float(rewards_all.mean().item())
    return metrics


def save_checkpoint(model, optimizer, path, global_step, episode):
    """保存训练checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'episode': episode,
        'model_type': 'history_transformer_ppo',
    }, path)


def _checkpoint_path(base_path, global_step):
    root, ext = os.path.splitext(base_path)
    return '%s_step_%d%s' % (root, global_step, ext or '.pt')


def _fmt_metric(value, digits=4):
    """格式化可能为空的指标"""
    if value is None:
        return 'N/A'
    return ('%%.%df' % digits) % value


def _print_train_log(episode, global_step, buffer_size, rewards,
                     decisions, stats, metrics):
    """打印足够详细但仍保持单行的训练日志"""
    hand_types = ','.join(
        '%s:%d' % (k, v) for k, v in stats['hand_types'].most_common(4))
    if not hand_types:
        hand_types = '-'
    print(
        'episode=%d step=%d decisions=%d buffer=%d updated=%s '
        'loss=%s policy=%s value=%s entropy=%s kl=%s clip=%s grad=%s '
        'adv_mean=%s adv_std=%s reward_mean=%s rewards=%s '
        'plays=%d passes=%d cha=%d dian=%d invalid=%d avg_actions=%.1f '
        'finished=%s order=%s hand_types=%s' % (
            episode, global_step, decisions, buffer_size,
            'Y' if metrics['updated'] else 'N',
            _fmt_metric(metrics['loss']),
            _fmt_metric(metrics['policy_loss']),
            _fmt_metric(metrics['value_loss']),
            _fmt_metric(metrics['entropy']),
            _fmt_metric(metrics['approx_kl']),
            _fmt_metric(metrics['clip_frac']),
            _fmt_metric(metrics['grad_norm']),
            _fmt_metric(metrics['adv_mean']),
            _fmt_metric(metrics['adv_std']),
            _fmt_metric(metrics['reward_mean']),
            ['%.2f' % r for r in rewards],
            stats['plays'], stats['passes'], stats['cha'], stats['dian'],
            stats['invalid'], stats['avg_candidates'],
            stats['finished'], stats['finish_order'], hand_types,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=0)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--update-steps', type=int, default=1024)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--debug-episodes', type=int, default=3)
    parser.add_argument('--save', default=os.path.join(
        ROOT, 'checkpoints', 'default_rl.pt'))
    args = parser.parse_args()

    if torch is None:
        raise RuntimeError(
            '训练需要安装 torch，请先执行: pip install -r requirements.txt')

    model = CardPolicyNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    buffer = []
    global_step = 0
    episode = 0
    next_save_step = args.save_every
    last_metrics = _empty_update_metrics(False)

    while True:
        if args.episodes and episode >= args.episodes:
            break
        if args.steps and global_step >= args.steps:
            break

        episode += 1
        transitions, rewards, decisions, stats = run_episode(model)
        buffer.extend(transitions)
        global_step += decisions

        metrics = _empty_update_metrics(False)
        if len(buffer) >= args.update_steps:
            metrics = ppo_update(model, optimizer, buffer)
            last_metrics = metrics
            buffer.clear()

        should_log = (
            episode <= args.debug_episodes
            or episode % args.log_every == 0
            or metrics['updated']
        )
        if should_log:
            effective_metrics = metrics if metrics['updated'] else last_metrics
            if not metrics['updated']:
                effective_metrics = dict(effective_metrics)
                effective_metrics['updated'] = False
            _print_train_log(
                episode, global_step, len(buffer), rewards,
                decisions, stats, effective_metrics)

        while global_step >= next_save_step:
            step_path = _checkpoint_path(args.save, next_save_step)
            save_checkpoint(model, optimizer, step_path,
                            next_save_step, episode)
            print('checkpoint已保存:', step_path)
            next_save_step += args.save_every

    if buffer:
        final_metrics = ppo_update(model, optimizer, buffer)
        print('final_update loss=%s policy=%s value=%s entropy=%s kl=%s clip=%s' % (
            _fmt_metric(final_metrics['loss']),
            _fmt_metric(final_metrics['policy_loss']),
            _fmt_metric(final_metrics['value_loss']),
            _fmt_metric(final_metrics['entropy']),
            _fmt_metric(final_metrics['approx_kl']),
            _fmt_metric(final_metrics['clip_frac']),
        ), flush=True)
    save_checkpoint(model, optimizer, args.save, global_step, episode)
    print('最终模型已保存到:', args.save)


if __name__ == '__main__':
    main()
