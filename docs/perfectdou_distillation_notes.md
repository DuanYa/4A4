# PerfectDou 风格蒸馏改造说明

## PerfectDou 的全局观察能力

PerfectDou 公开仓库只包含评估代码、预训练模型和部分 `.so` 特征模块，训练系统没有开源。仓库 README 明确说明训练代码不可用，核心思路是
perfect information distillation：训练阶段使用全局信息指导策略，实际对局时仍只使用不完全信息观测。

从公开评估代码能看到两条线索：

- DouZero baseline 仍是逐合法动作打分：历史 `z_batch` 进 LSTM，局面和动作特征 `x_batch` 拼接后过多层 MLP 输出每个候选动作的 value。
- PerfectDou 推理模型是 ONNX 全局 action id 分类式输出，输入包含不带动作的局面特征和合法动作数组，再选 `argmax(action_logit)`，最后解码为当前手牌里的实际动作。

PerfectDou 的“全局观察”不是线上作弊。它的关键是训练期 teacher 可以看到完整牌局，从而提供更低方差、更快收敛的监督信号；上线 policy 则被蒸馏到只依赖自身可见信息。

## 当前 4A4 改造

本分支保留线上执行路径：

- `rl.policy.NeuralPolicy` 仍然只调用 `model(state, action_ids, history)`。
- 服务器和 AI 玩家仍然收到物理 `indices`。
- 全局信息只在训练 transition 中保存，并且只输入 Perfect Critic，不进入 Actor 和后端接口。

新增训练期能力：

- `encode_perfect_state(game, seat, state)` 编码四家手牌、各家已出牌、当前阶段、台上队伍、最后牌型等 full-info 信号。
- `CardPolicyNetwork.forward_actor_critic(..., perfect_state_vec=...)` 中 Actor 仍使用不完全信息，Critic 在训练时使用 full-info encoder。
- `dmc_update(...)` 保留函数名兼容现有分布式脚本，但实际执行 PPO clipped objective。
- episode 结束后按每个座位轨迹用 GAE 计算 advantage/return。
- reward 由终局团队收益 + 胜利距离变化组成。

## 为什么预期更快收敛

4A4 是不完全信息团队游戏，单纯用终局 reward 训练 value，很多状态下同一可见观测对应多种隐藏手牌配置，目标方差很大。full-info critic 直接看到隐藏手牌，可以更稳定地估计 advantage；Actor 仍被约束在真实可见观测上，因此训练信息不会泄漏到执行期。

这与 PerfectDou 的 perfect-training / imperfect-execution 框架一致，但没有引入线上全局信息，因此不会破坏真实对局约束。
