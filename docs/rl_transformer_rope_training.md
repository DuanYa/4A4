# 4A4 强化学习训练说明：RoPE Transformer 版

## 本次结构变化

- 历史出牌长度从 40 条扩展到 80 条。
- 历史 Transformer 使用 RoPE 旋转位置编码。
- 全零历史占位会被 mask 掉，不参与 attention，也不参与历史池化。
- 训练主路径去掉 LSTM teacher/student 双模型，直接训练后端可加载的 `rl.model.CardPolicyNetwork`。
- 四个座位共享同一个模型。当前特征已经按出牌者视角编码相对座位、队友、对手、自身等信息，4A4 对称对弈下共享模型可以合并四个座位样本，通常比四套独立 LSTM 更省样本、更容易上线。

## 关键文件

- `rl/model.py`：后端推理模型与训练模型，同一个 RoPE Transformer。
- `rl/features.py`：历史长度跟随 `HISTORY_LEN=80`，padding 行为全零。
- `scripts/train_douzero_4a4.py`：单机 DMC 训练。
- `scripts/train_douzero_4a4_distributed.py`：多进程 actor + 单 learner/replay buffer 分布式训练。

## 远端训练推荐命令

远端目录：

```bash
cd /home/dyeea/4A4
```

进入环境：

```bash
source /home/dyeea/miniconda3/etc/profile.d/conda.sh
conda activate faf
```

启动分布式训练：

```bash
mkdir -p logs checkpoints
tmux new-session -d -s 4a4-rl-rope \
  "bash -lc 'cd /home/dyeea/4A4 && \
  source /home/dyeea/miniconda3/etc/profile.d/conda.sh && \
  conda activate faf && \
  export PYTHONUNBUFFERED=1 && \
  python scripts/train_douzero_4a4_distributed.py \
    --device cuda:0 \
    --actor-devices cuda:0,cuda:1,cuda:2 \
    --actors-per-device 4 \
    --frames 100000000 \
    --batch-size 512 \
    --replay-size 80000 \
    --min-replay 8192 \
    --update-frames 4096 \
    --update-batches 8 \
    --checkpoint-every 1000000 \
    --eval-every 5000000 \
    --eval-episodes 256 \
    --log-every 100000 \
    --save checkpoints/douzero_4a4_rope_transformer.pt \
    --log-jsonl logs/douzero_4a4_rope_transformer_metrics.jsonl \
    2>&1 | tee -a logs/douzero_4a4_rope_transformer_train.log'"
```

查看实时日志：

```bash
tmux attach -t 4a4-rl-rope
```

或：

```bash
tail -f /home/dyeea/4A4/logs/douzero_4a4_rope_transformer_train.log
```

## 日志指标怎么看

- `样本步数`：已用于训练/回放的决策样本数，越高代表训练越充分。
- `局数`：采样完成的对局数。
- `每秒样本`：采样吞吐，越高越好。
- `回放池样本`：replay buffer 当前样本数，达到 `min-replay` 后才会稳定更新。
- `采样队列`：actor 往 learner 投递的队列积压。长期为 0 可能 learner 等数据；长期满可能 learner 太慢。
- `训练损失`：DMC Q 回归 MSE，不是越低绝对越好，但同一配置下应逐步稳定。
- `平均绝对误差`：预测 Q 和终局收益的平均差，越低通常越好。
- `平均Q值`：模型当前对所选动作的价值估计，主要观察是否发散。
- `梯度范数`：更新梯度大小，长期极大说明训练不稳。
- `正常结束率`：未被最大步数截断的对局比例，越高越好。
- `全洞率`、`半洞率`：牌局结果结构，用于观察策略是否偏激或异常。
- `台上切换率`：台上队伍变化比例。
- `非法动作数`：应尽量接近 0。
- `平均候选动作数`：每次决策可选动作复杂度。
- `平均所选Q值`、`平均贪心Q值`、`平均探索Q差`：探索动作和贪心动作的价值差，差距过大说明探索代价高。
- `过牌率`：过牌占比，过高可能保守，过低可能乱压牌。
- `探索动作率`：epsilon 探索实际比例。
- `不叉率`、`不点率`：叉/点策略倾向，过高或过低都需要结合胜率看。
- `评估胜率`：模型队伍对规则队伍的评估胜率，越高越好，是选 checkpoint 的核心指标。

## 产物

最终 checkpoint：

```text
checkpoints/douzero_4a4_rope_transformer.pt
```

中间 checkpoint：

```text
checkpoints/douzero_4a4_rope_transformer_step_*.pt
```

结构化中文指标：

```text
logs/douzero_4a4_rope_transformer_metrics.jsonl
```

## 注意事项

- 本次模型结构改变后，旧 checkpoint 只能部分兼容加载，建议从新结构重新训练。
- checkpoint 文件不随代码同步覆盖远端；训练会在远端本地生成。
- 后端加载模型时需要同步本次 `rl/model.py` 和 `rl/features.py`，否则 checkpoint 维度不匹配。
