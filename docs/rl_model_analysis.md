# 四幺四(4A4) - 强化学习模型结构与训练流程分析文档

> 所有流程图、架构图、UML图均使用 Mermaid 格式

---

## 目录

1. [整体概览：两套RL系统](#1-整体概览两套rl系统)
2. [CardPolicyNetwork (PPO Actor-Critic) 详细结构](#2-cardpolicynetwork-ppo-actor-critic-详细结构)
3. [DouZero4A4Model (DMC LSTM-Q) 详细结构](#3-douzero4a4model-dmc-lstm-q-详细结构)
4. [特征编码系统详解](#4-特征编码系统详解)
5. [动作空间与合法动作枚举](#5-动作空间与合法动作枚举)
6. [PPO训练流程详解](#6-ppo训练流程详解)
7. [DMC训练流程详解](#7-dmc训练流程详解)
8. [奖励系统详解](#8-奖励系统详解)
9. [自对弈与模型共享](#9-自对弈与模型共享)
10. [模型部署与推理](#10-模型部署与推理)
11. [函数清单](#11-函数清单)

---

## 1. 整体概览：两套RL系统

### 1.1 系统对比

```mermaid
graph TB
    subgraph PPO["PPO系统 (train_rl_ai.py)"]
        PPO_MODEL["CardPolicyNetwork<br/>共享状态编码器 +<br/>Transfomer历史编码器 +<br/>动作编码器 +<br/>Actor头 + Critic头"]
        PPO_ENV["PPOEnvironment<br/>封装4个AI self-play<br/>综合 team_reward"]
        PPO_ALGO["PPO算法<br/>Clip Epsilon=0.2<br/>GAE(lambda=0.95)<br/>Entropy Coef=0.01<br/>Replay Buffer"]
        PPO_TRAIN["训练循环<br/>采集→计算优势→多epoch更新<br/>每N步更新target<br/>每M步保存checkpoint"]
    end

    subgraph DMC["DMC系统 (train_douzero_4a4.py)"]
        DMC_MODEL["DouZero4A4Model<br/>4个独立LSTM<br/>共享Q头部<br/>位置感知预测"]
        DMC_ENV["Simulation环境<br/>MCTS collect<br/>或随机rollout"]
        DMC_ALGO["DMC算法<br/>终端团队奖励<br/>MSE loss<br/>LSTM状态管理"]
        DMC_TRAIN["训练循环<br/>回合生成→reward计算<br/>→batch更新→多GPU"]
    end

    subgraph SHARED["共享组件"]
        FEAT["features.py<br/>encode_state(160)<br/>encode_actions(N,48)<br/>encode_history(40,64)"]
        ACTS["actions.py<br/>enumerate_legal_actions<br/>turn_actions<br/>cha_actions<br/>dian_actions"]
        STORE["model_store.py<br/>共享模型单例缓存<br/>get_shared_model()<br/>resolve_model_path()"]
    end

    PPO_ENV --> FEAT
    PPO_ENV --> ACTS
    DMC_ENV --> FEAT
    DMC_ENV --> ACTS
    PPO_MODEL --> FEAT
    DMC_MODEL --> FEAT
```

### 1.2 两套系统架构对比

| 维度 | PPO (CardPolicyNetwork) | DMC (DouZero4A4Model) |
|------|------------------------|----------------------|
| **算法类型** | PPO Actor-Critic | Deep Monte Carlo (Q-learning) |
| **网络结构** | 共享编码器 + Transformer + Actor/Critic头 | 4个独立LSTM + 共享Q头 |
| **输出** | 动作概率分布 + 状态价值 | 每个动作的Q值 |
| **探索方式** | 策略熵 + epsilon-greedy fallback | epsilon-greedy |
| **历史建模** | Transformer (40步历史) | LSTM隐藏状态 |
| **位置感知** | 否（统一模型） | 是（4个位置独立LSTM） |
| **奖励** | 即时团队奖励 + 终局奖励 | 仅终局团队奖励 |
| **训练效率** | 高（采样效率高） | 低（需要完整回合） |
| **参考来源** | 自研PPO实现 | 参考DouZero (Kwai) |

---

## 2. CardPolicyNetwork (PPO Actor-Critic) 详细结构

### 2.1 网络架构图

```mermaid
graph TB
    subgraph INPUT["输入"]
        STATE["state_vec<br/>[B, 160]<br/>当前局面特征"]
        HISTORY["history_vecs<br/>[B, 40, 64]<br/>最近40步历史"]
        ACTIONS["action_vecs<br/>[B, N_actions, 48]<br/>N个候选动作"]
    end

    subgraph STATE_ENCODER["状态编码器 (MLP)"]
        S_FC1["Linear(160, 256)"]
        S_BN1["BatchNorm1d(256)"]
        S_RELU1["ReLU"]
        S_FC2["Linear(256, 256)"]
        S_BN2["BatchNorm1d(256)"]
        S_RELU2["ReLU"]
        S_OUT["state_encoded<br/>[B, 256]"]
    end

    subgraph ACT_ENCODER["动作编码器 (MLP, 共享)"]
        A_FC1["Linear(48, 64)"]
        A_RELU1["ReLU"]
        A_FC2["Linear(64, 64)"]
        A_RELU2["ReLU"]
        A_OUT["action_encoded<br/>[B, N, 64]"]
    end

    subgraph HIST_ENCODER["历史编码器 (Transformer)"]
        H_POS["PositionalEncoding<br/>[B, 40, 64]"]
        H_LAYER1["TransformerEncoderLayer<br/>d_model=64, nhead=4<br/>dim_feedforward=128"]
        H_LAYER2["TransformerEncoderLayer<br/>d_model=64, nhead=4<br/>dim_feedforward=128"]
        H_POOL{"平均池化<br/>[B, 40, 64] → [B, 64]"}
    end

    subgraph FUSION["特征融合"]
        F_CONCAT["torch.cat<br/>[state_256, action_64, hist_64]<br/>→ [B, N, 384]"]
    end

    subgraph ACTOR["Actor头 (策略网络)"]
        AC_FC1["Linear(384, 256)"]
        AC_RELU1["ReLU"]
        AC_FC2["Linear(256, 128)"]
        AC_RELU2["ReLU"]
        AC_FC3["Linear(128, 1)"]
        AC_SQUEEZE["squeeze(-1) → [B, N]"]
        AC_LOGITS["logits [B, N]<br/>每个动作的logit"]
    end

    subgraph CRITIC["Critic头 (价值网络)"]
        CR_MEAN["state_encoded.unsqueeze(1)<br/>→ [B, 1, 256]"]
        CR_EXPAND["expand → [B, N, 256]"]
        CR_LINEAR["Linear(256, 1)"]
        CR_SQUEEZE["squeeze(-1) → [B, N, 1]"]
        CR_MEAN2["mean(dim=1) → [B, 1]"]
        CR_OUT["value [B, 1]<br/>状态价值估计"]
    end

    STATE --> STATE_ENCODER
    HISTORY --> HIST_ENCODER
    ACTIONS --> ACT_ENCODER
    S_OUT --> F_CONCAT
    H_POOL --> F_CONCAT
    A_OUT --> F_CONCAT
    F_CONCAT --> ACTOR
    F_CONCAT --> CRITIC
```

### 2.2 数据流维度变换表

| 层 | 输入维度 | 输出维度 | 操作 |
|----|---------|---------|------|
| `state_vec` | `[B, 160]` | `[B, 160]` | 原始输入 |
| `Linear(160→256)` | `[B, 160]` | `[B, 256]` | 全连接 |
| `BatchNorm1d(256)` | `[B, 256]` | `[B, 256]` | 批归一化 |
| `ReLU` | `[B, 256]` | `[B, 256]` | 激活 |
| `Linear(256→256)` | `[B, 256]` | `[B, 256]` | 全连接 |
| `BatchNorm1d(256)` | `[B, 256]` | `[B, 256]` | 批归一化 |
| `ReLU` | `[B, 256]` | `[B, 256]` | 激活 |
| `history_vecs` | `[B, 40, 64]` | `[B, 40, 64]` | 原始输入 |
| `PositionalEncoding` | `[B, 40, 64]` | `[B, 40, 64]` | 位置编码(加法) |
| `TransformerLayer×2` | `[B, 40, 64]` | `[B, 40, 64]` | 自注意力 |
| `mean(dim=1)` | `[B, 40, 64]` | `[B, 64]` | 平均池化 |
| `action_vecs` | `[B, N, 48]` | `[B, N, 48]` | 原始输入 |
| `Linear(48→64)` | `[B, N, 48]` | `[B, N, 64]` | 全连接(共享) |
| `ReLU` | `[B, N, 64]` | `[B, N, 64]` | 激活 |
| `Linear(64→64)` | `[B, N, 64]` | `[B, N, 64]` | 全连接(共享) |
| `ReLU` | `[B, N, 64]` | `[B, N, 64]` | 激活 |
| `torch.cat` | `[B,256]+[B,N,64]+[B,64]` | `[B, N, 384]` | 拼接(广播) |
| `Linear(384→256)` | `[B, N, 384]` | `[B, N, 256]` | Actor全连接 |
| `ReLU` | `[B, N, 256]` | `[B, N, 256]` | 激活 |
| `Linear(256→128)` | `[B, N, 256]` | `[B, N, 128]` | Actor全连接 |
| `ReLU` | `[B, N, 128]` | `[B, N, 128]` | 激活 |
| `Linear(128→1)` | `[B, N, 128]` | `[B, N, 1]` | Actor输出 |
| `squeeze(-1)` | `[B, N, 1]` | `[B, N]` | 压缩维度 |
| **最终输出** | | | |
| `logits` | | `[B, N]` | 每个动作的logit |
| `value` | | `[B, 1]` | 状态价值估计 |

### 2.3 forward_actor_critic 详细流程

```mermaid
flowchart TD
    F1["forward_actor_critic(state_vec, action_vecs, history_vecs)"] --> F2["state_enc = state_encoder(state_vec)<br/>[B,160] → [B,256]"]
    F2 --> F3["action_enc = action_encoder(action_vecs)<br/>[B,N,48] → [B,N,64]"]
    F3 --> F4["hist_enc = history_encoder(history_vecs)<br/>[B,40,64] → [B,64]"]
    F4 --> F5["action_enc_exp = action_enc<br/>state_enc_exp = state_enc.unsqueeze(1).expand(-1,N,-1)<br/>hist_enc_exp = hist_enc.unsqueeze(1).expand(-1,N,-1)"]
    F5 --> F6["combined = cat([state_enc_exp, action_enc_exp, hist_enc_exp], dim=-1)<br/>[B,N,384]"]
    F6 --> F7["logits = actor_head(combined).squeeze(-1)<br/>[B,N,384] → [B,N]"]
    F7 --> F8["value = critic_head(state_enc)<br/>[B,256] → [B,1]"]
    F8 --> F9["返回 (logits, value)"]
```

### 2.4 __init__ 构造函数参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `state_dim` | 160 | 状态特征向量维度 |
| `action_dim` | 48 | 单个动作特征向量维度 |
| `history_dim` | 64 | 历史每一步特征向量维度 |
| `history_len` | 40 | 保留的历史步数 |
| `hidden_dim` | 256 | 隐藏层维度 |
| `nhead` | 4 | Transformer多头注意力头数 |
| `num_transformer_layers` | 2 | Transformer编码器层数 |

---

## 3. DouZero4A4Model (DMC LSTM-Q) 详细结构

### 3.1 网络架构图

```mermaid
graph TB
    subgraph INPUT["输入"]
        STATE["obs<br/>[B, state_dim]"]
        POS["position<br/>[B, 1] 位置ID (0-3)"]
        ACTIONS["action_features<br/>[B, N_actions, action_dim]"]
    end

    subgraph LSTM_MODULES["4个位置独立LSTM模块"]
        LSTM0["LSTM0<br/>LSTM(256, 512)<br/>Linear(512→512)"]
        LSTM1["LSTM1<br/>LSTM(256, 512)<br/>Linear(512→512)"]
        LSTM2["LSTM2<br/>LSTM(256, 512)<br/>Linear(512→512)"]
        LSTM3["LSTM3<br/>LSTM(256, 512)<br/>Linear(512→512)"]
    end

    subgraph MLP["共享MLP编码器"]
        M_FC1["Linear(state_dim, 256)"]
        M_RELU1["ReLU"]
        M_FC2["Linear(256, 256)"]
        M_RELU2["ReLU"]
        M_OUT["encoded_state<br/>[B, 256]"]
    end

    subgraph Q_HEAD["Q值头 (共享)"]
        Q_FC1["Linear(512, 256)"]
        Q_RELU1["ReLU"]
        Q_FC2["Linear(256, 128)"]
        Q_RELU2["ReLU"]
        Q_FC3["Linear(128, 1)"]
        Q_OUT["q_values<br/>[B, N_actions]<br/>每个动作的Q值"]
    end

    STATE --> MLP
    M_OUT --> LSTM0
    M_OUT --> LSTM1
    M_OUT --> LSTM2
    M_OUT --> LSTM3
    POS --> LSTM0
    POS --> LSTM1
    POS --> LSTM2
    POS --> LSTM3
    LSTM0 --> Q_HEAD
    LSTM1 --> Q_HEAD
    LSTM2 --> Q_HEAD
    LSTM3 --> Q_HEAD
```

### 3.2 forward 详细流程

```mermaid
flowchart TD
    F1["forward(obs, position, action_features, lstm_state=None)"] --> F2["x = MLP(obs)<br/>[B,state_dim] → [B,256]"]
    F2 --> F3["x = x.unsqueeze(0)<br/>[B,256] → [1,B,256]"]
    F3 --> F4{"position = ?"}
    F4 -->|0| F5["lstm_out, new_state = lstm0(x, lstm_state)"]
    F4 -->|1| F6["lstm_out, new_state = lstm1(x, lstm_state)"]
    F4 -->|2| F7["lstm_out, new_state = lstm2(x, lstm_state)"]
    F4 -->|3| F8["lstm_out, new_state = lstm3(x, lstm_state)"]
    F5 --> F9["lstm_out = lstm_out[0]<br/>[1,B,512] → [B,512]"]
    F6 --> F9
    F7 --> F9
    F8 --> F9
    F9 --> F10["lstm_out = lstm_out.unsqueeze(1).expand(-1, N_actions, -1)<br/>[B,512] → [B,N_actions,512]"]
    F10 --> F11["q_values = Q_head(lstm_out)<br/>[B,N_actions,512] → [B,N_actions]"]
    F11 --> F12["返回 (q_values, new_lstm_state)"]
```

### 3.3 数据流维度变换表

| 层 | 输入维度 | 输出维度 | 操作 |
|----|---------|---------|------|
| `obs` | `[B, state_dim]` | `[B, state_dim]` | 原始输入 |
| `Linear(state_dim→256)` | `[B, state_dim]` | `[B, 256]` | 全连接 |
| `ReLU` | `[B, 256]` | `[B, 256]` | 激活 |
| `Linear(256→256)` | `[B, 256]` | `[B, 256]` | 全连接 |
| `ReLU` | `[B, 256]` | `[B, 256]` | 激活 |
| `unsqueeze(0)` | `[B, 256]` | `[1, B, 256]` | 添加时间步 |
| `LSTM(256→512)` | `[1, B, 256]` | `[1, B, 512]` | LSTM前向 |
| `squeeze(0)` | `[1, B, 512]` | `[B, 512]` | 移除时间步 |
| `unsqueeze(1)+expand` | `[B, 512]` | `[B, N, 512]` | 广播 |
| `Linear(512→256)` | `[B, N, 512]` | `[B, N, 256]` | Q头全连接 |
| `ReLU` | `[B, N, 256]` | `[B, N, 256]` | 激活 |
| `Linear(256→128)` | `[B, N, 256]` | `[B, N, 128]` | Q头全连接 |
| `ReLU` | `[B, N, 128]` | `[B, N, 128]` | 激活 |
| `Linear(128→1)` | `[B, N, 128]` | `[B, N, 1]` | Q头输出 |
| `squeeze(-1)` | `[B, N, 1]` | `[B, N]` | 压缩维度 |
| **最终输出** | | | |
| `q_values` | | `[B, N]` | 每个动作的Q值 |
| `lstm_state` | | `(h, c)` | LSTM隐藏状态元组 |

### 3.4 位置感知LSTM机制

```mermaid
flowchart LR
    subgraph "同一局游戏的4个玩家"
        P0["玩家0<br/>LSTM0状态"]
        P1["玩家1<br/>LSTM1状态"]
        P2["玩家2<br/>LSTM2状态"]
        P3["玩家3<br/>LSTM3状态"]
    end

    subgraph "LSTM状态流转"
        S0["初始状态<br/>h=0, c=0"] -->|"玩家0出牌"| P0
        P0 -->|"返回new_state"| S0
        S0 -->|"玩家1出牌"| P1
        P1 -->|"返回new_state"| S1
    end

    P0 -.->|"独立权重"| LSTM0["LSTM0<br/>encoder + LSTM"]
    P1 -.->|"独立权重"| LSTM1["LSTM1<br/>encoder + LSTM"]
    P2 -.->|"独立权重"| LSTM2["LSTM2<br/>encoder + LSTM"]
    P3 -.->|"独立权重"| LSTM3["LSTM3<br/>encoder + LSTM"]
```

---

## 4. 特征编码系统详解

### 4.1 状态编码 (encode_state)

```mermaid
flowchart TD
    ENC["encode_state(hand, level_rank, last_play, played_cards,<br/>other_hand_sizes, other_finished, finish_order,<br/>is_free_play, current_seat, phase)"] --> F1["创建零向量 → [160]"]

    F1 --> S1["手牌编码 [0:52]<br/>每位表示一张牌是否有"]
    S1 --> S2["级牌标记 [52:56]<br/>4位：是否级牌点数"]
    S2 --> S3["手牌统计 [56:60]<br/>4位：手牌数量分布"]
    S3 --> S4["其他玩家手牌数 [60:64]<br/>4位：每人手牌数/16"]
    S4 --> S5["完成状态 [64:68]<br/>4位：每人是否完成"]
    S5 --> S6["完成顺序 [68:72]<br/>4位：每人完成名次"]
    S6 --> S7["当前出牌者 [72:76]<br/>4位：one-hot(seat)"]
    S7 --> S8["自由出牌 [76]<br/>1位：is_free_play"]
    S8 --> S9["游戏阶段 [77:80]<br/>3位：one-hot(phase)"]
    S9 --> S10["场上最后出牌类型 [80:88]<br/>8位：one-hot(hand_type)"]
    S10 --> S11["最后出牌点数 [88:100]<br/>12位：点数编码"]
    S11 --> S12["最后出牌座位 [100:104]<br/>4位：one-hot(seat)"]
    S12 --> S13["已出牌统计 [104:156]<br/>52位：每张牌是否已出"]
    S13 --> S14["级牌点数值 [156:160]<br/>4位：level_rank索引"]

    S14 --> RET["返回 [160] float32 tensor"]
```

### 4.2 动作编码 (encode_action)

```mermaid
flowchart TD
    AENC["encode_action(action_indices, hand_size, is_free_play, last_ht)"] --> F1["创建零向量 → [48]"]

    F1 --> A1["动作类型 [0:8]<br/>8位：one-hot(hand_type)"]
    A1 --> A2["出牌点数 [8:20]<br/>12位：点数编码"]
    A2 --> A3["手牌占比 [20:28]<br/>8位：手牌使用比例"]
    A3 --> A4["是否pass [28]<br/>1位"]
    A4 --> A5["是否free_play [29]<br/>1位"]
    A5 --> A6["是否出完 [30]<br/>1位：出完后手牌=0"]
    A6 --> A7["可管住上家 [31:39]<br/>8位：can_beat每类"]
    A7 --> A8["剩余手牌分析 [39:48]<br/>9位：剩余牌型统计"]

    A8 --> RET["返回 [48] float32 tensor"]
```

### 4.3 历史编码 (encode_history)

```mermaid
flowchart TD
    HENC["encode_history(play_history, max_len=40)"] --> F1["创建零矩阵 → [40, 64]"]

    F1 --> H1["对每个历史步 (最多40步):"]
    H1 --> H2["动作类型 [0:8]<br/>8位：one-hot(play/cha/dian/pass)"]
    H2 --> H3["出牌座位 [8:12]<br/>4位：one-hot(seat)"]
    H3 --> H4["出牌点数 [12:24]<br/>12位：点数编码"]
    H4 --> H5["牌型类别 [24:32]<br/>8位：one-hot(hand_type)"]
    H5 --> H6["手牌减少量 [32:40]<br/>8位：减少比例"]
    H6 --> H7["团队信息 [40:44]<br/>4位：是否是队友"]
    H7 --> H8["对手信息 [44:48]<br/>4位：是否是对手"]
    H8 --> H9["是否引发叉 [48]<br/>1位"]
    H9 --> H10["是否被叉 [49]<br/>1位"]
    H10 --> H11["是否引发点 [50]<br/>1位"]
    H11 --> H12["自由出牌 [51]<br/>1位"]
    H12 --> H13["其他维度 [52:64]<br/>12位：保留"]

    H13 --> RET["返回 [40, 64] float32 tensor"]
```

### 4.4 函数清单

| 函数 | 位置 | 签名 | 职责 |
|------|------|------|------|
| `encode_state(hand, level_rank, last_play, played_cards, other_hand_sizes, other_finished, finish_order, is_free_play, current_seat, phase)` | rl/features.py | `def encode_state(...)` | 编码当前游戏局面为160维向量 |
| `encode_action(action_indices, hand_size, is_free_play, last_ht)` | rl/features.py | `def encode_action(...)` | 编码单个候选动作→48维向量 |
| `encode_actions(action_list, hand_size, is_free_play, last_ht)` | rl/features.py | `def encode_actions(...)` | 批量编码候选动作→[N,48] |
| `encode_history(play_history, max_len=40)` | rl/features.py | `def encode_history(play_history, max_len=40)` | 编码出牌历史→[40,64] |
| `_encode_cards_onehot(cards)` | rl/features.py | `def _encode_cards_onehot(cards)` | 手牌→52位one-hot编码 |
| `_encode_rank(rank)` | rl/features.py | `def _encode_rank(rank)` | 点数→12位编码 |
| `_encode_hand_type(ht)` | rl/features.py | `def _encode_hand_type(ht)` | 牌型→8位one-hot编码 |
| `_encode_phase(phase)` | rl/features.py | `def _encode_phase(phase)` | 游戏阶段→3位one-hot编码 |

---

## 5. 动作空间与合法动作枚举

### 5.1 动作枚举完整流程

```mermaid
flowchart TD
    ENUM["enumerate_legal_actions(hand, level_rank, last_ht, is_free_play)"] --> C1{"is_free_play?"}
    C1 -->|是| FREE["枚举所有自由出牌动作"]
    C1 -->|否| BEAT["枚举所有跟牌动作"]

    FREE --> F1["枚举单张"] --> F2["枚举对子"] --> F3["枚举单龙"]
    F3 --> F4["枚举双龙"] --> F5["枚举炸弹3"] --> F6["枚举炸弹4"]
    F6 --> F7["枚举双王"] --> F8["枚举四幺四"]
    F8 --> F9["添加pass(自由出牌时不可pass)"]

    BEAT --> B1{"last_ht 类型?"}
    B1 -->|SINGLE| B2["枚举所有更大单张 + 炸弹"]
    B1 -->|PAIR| B3["枚举所有更大对子 + 炸弹"]
    B1 -->|STRAIGHT| B4["枚举同长度更大单龙 + 双龙 + 炸弹"]
    B1 -->|DOUBLE_STRAIGHT| B5["枚举同长度更大双龙 + 炸弹"]
    B1 -->|BOMB3| B6["枚举更大炸弹3 + 炸弹4 + 双王 + 四幺四"]
    B1 -->|BOMB4| B7["枚举更大炸弹4 + 双王 + 四幺四"]
    B1 -->|JOKER_BOMB| B8["枚举四幺四"]
    B1 -->|SI_YAO_SI| B9["无可管牌"]
    B9 --> B10["添加pass"]

    F9 --> RET["返回 [(card_indices, hand_type), ...]"]
    B10 --> RET
```

### 5.2 动作空间大小分析

| 场景 | 典型动作数 | 最小 | 最大 |
|------|-----------|------|------|
| 自由出牌（满手牌） | 15-25 | 5 | 40+ |
| 跟牌（大牌在手） | 5-10 | 1 | 20+ |
| 跟牌（无牌可出） | 1 | 1 | 1 |
| 平均每步 | 8-12 | 1 | 40+ |

### 5.3 函数清单

| 函数 | 位置 | 签名 | 职责 |
|------|------|------|------|
| `enumerate_legal_actions(hand, level_rank, last_ht, is_free_play)` | rl/actions.py | `def enumerate_legal_actions(...)` | 枚举所有合法动作（出牌+pass） |
| `_enumerate_free_actions(hand, level_rank)` | rl/actions.py | `def _enumerate_free_actions(...)` | 枚举自由出牌时所有合法出牌 |
| `_enumerate_beat_actions(hand, level_rank, last_ht)` | rl/actions.py | `def _enumerate_beat_actions(...)` | 枚举跟牌时所有合法出牌 |
| `_enumerate_singles(hand)` | rl/actions.py | `def _enumerate_singles(hand)` | 枚举所有单张出牌 |
| `_enumerate_pairs(hand)` | rl/actions.py | `def _enumerate_pairs(hand)` | 枚举所有对子出牌 |
| `_enumerate_straights(hand, level_rank)` | rl/actions.py | `def _enumerate_straights(...)` | 枚举所有单龙出牌 |
| `_enumerate_double_straights(hand, level_rank)` | rl/actions.py | `def _enumerate_double_straights(...)` | 枚举所有双龙出牌 |
| `_enumerate_bombs3(hand)` | rl/actions.py | `def _enumerate_bombs3(hand)` | 枚举所有炸弹3出牌 |
| `_enumerate_bombs4(hand)` | rl/actions.py | `def _enumerate_bombs4(hand)` | 枚举所有炸弹4出牌 |
| `_enumerate_joker_bomb(hand)` | rl/actions.py | `def _enumerate_joker_bomb(hand)` | 检查双王出牌 |
| `_enumerate_si_yao_si(hand)` | rl/actions.py | `def _enumerate_si_yao_si(hand)` | 检查四幺四出牌 |

---

## 6. PPO训练流程详解

### 6.1 训练主循环

```mermaid
flowchart TD
    START(["train_rl_ai.py main()"]) --> INIT["初始化"]
    INIT --> I1["创建 CardPolicyNetwork"]
    I1 --> I2["创建 optimizer (Adam, lr=3e-4)"]
    I2 --> I3["创建 PPOEnvironment (4个AI)"]
    I3 --> I4["加载 checkpoint (如果有)"]

    I4 --> LOOP["训练循环 (for episode in range(max_episodes))"]
    LOOP --> COLLECT["采集阶段"]
    COLLECT --> C1["env.reset()"]
    C1 --> C2["for step in range(horizon):"]
    C2 --> C3["对每个活跃AI并行推理"]
    C3 --> C4["获取 logits, value"]
    C4 --> C5["计算概率分布 softmax(logits)"]
    C5 --> C6["采样动作 (categorical)"]
    C6 --> C7["env.step(action) → reward, next_state, done"]
    C7 --> C8["存入 replay buffer"]
    C8 --> C9{"done?"}
    C9 -->|否| C2
    C9 -->|是| UPDATE["更新阶段"]

    UPDATE --> U1["计算 GAE 优势估计"]
    U1 --> U2["计算 returns = advantages + values"]
    U2 --> U3["for epoch in range(ppo_epochs):"]
    U3 --> U4["从 buffer 抽样 mini-batch"]
    U4 --> U5["前向传播: logits, values"]
    U5 --> U6["计算新概率分布"]
    U6 --> U7["计算 ratio = π_new / π_old"]
    U7 --> U8["计算 clipped surrogate objective"]
    U8 --> U9["计算 value_loss = MSE(returns, values)"]
    U9 --> U10["计算 entropy_loss"]
    U10 --> U11["total_loss = policy_loss + vf_coef*value_loss - ent_coef*entropy"]
    U11 --> U12["反向传播 + optimizer.step()"]
    U12 --> U3

    U12 --> LOG["记录日志"]
    LOG --> SAVE{"每 save_interval 步?"}
    SAVE -->|是| S1["保存 checkpoint"]
    SAVE -->|否| LOOP
    S1 --> LOOP
```

### 6.2 PPO损失函数详解

```mermaid
flowchart TD
    subgraph POLICY_LOSS["策略损失: Clipped Surrogate Objective"]
        PL1["ratio = π_new(a|s) / π_old(a|s)"] --> PL2["surr1 = ratio * advantage"]
        PL1 --> PL3["surr2 = clip(ratio, 1-ε, 1+ε) * advantage"]
        PL2 --> PL4["policy_loss = -min(surr1, surr2).mean()"]
        PL3 --> PL4
    end

    subgraph VALUE_LOSS["价值损失: MSE"]
        VL1["value_loss = F.mse_loss(values, returns)"]
    end

    subgraph ENTROPY["熵奖励: 鼓励探索"]
        EL1["entropy = -(probs * log(probs)).sum(-1).mean()"]
    end

    PL4 --> TOTAL["total_loss = policy_loss + vf_coef*value_loss - ent_coef*entropy"]
    VL1 --> TOTAL
    EL1 --> TOTAL
```

### 6.3 GAE优势估计

```mermaid
flowchart TD
    GAE1["GAE(λ=0.95, γ=0.99)"] --> G1["对于每个时间步 t (从后往前):"]
    G1 --> G2["δ_t = r_t + γ*V(s_{t+1}) - V(s_t)"]
    G2 --> G3["A_t = δ_t + γ*λ*A_{t+1}"]
    G3 --> G4["returns_t = A_t + V(s_t)"]
    G4 --> G5["返回 advantages, returns"]
```

### 6.4 PPO超参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `lr` | 3e-4 | 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE λ参数 |
| `clip_epsilon` | 0.2 | PPO裁剪范围 |
| `ppo_epochs` | 10 | 每次更新迭代次数 |
| `batch_size` | 64 | mini-batch大小 |
| `horizon` | 2048 | 每轮采集步数 |
| `vf_coef` | 0.5 | 价值损失系数 |
| `ent_coef` | 0.01 | 熵系数 |
| `max_grad_norm` | 0.5 | 梯度裁剪阈值 |

---

## 7. DMC训练流程详解

### 7.1 训练主循环

```mermaid
flowchart TD
    START(["train_douzero_4a4.py main()"]) --> INIT["初始化"]
    INIT --> I1["创建 DouZero4A4Model"]
    I1 --> I2["创建 optimizer (Adam, lr=1e-4)"]
    I2 --> I3["加载 checkpoint (如果有)"]
    I3 --> I4["如果多GPU: DataParallel(model)"]

    I4 --> LOOP["训练循环 (for episode in range(max_episodes))"]
    LOOP --> GEN["回合生成"]
    GEN --> G1["创建新 Game 环境"]
    G1 --> G2["while not game_over:"]
    G2 --> G3["获取当前玩家 state"]
    G3 --> G4["获取当前玩家 position"]
    G4 --> G5["enumerate_legal_actions()"]
    G5 --> G6{"epsilon-greedy?"}
    G6 -->|是| G7["随机选择合法动作"]
    G6 -->|否| G8["model.forward(state, position, actions, lstm_state)"]
    G8 --> G9["q_values = model output"]
    G9 --> G10["action = argmax(q_values)"]
    G7 --> G11["执行动作 → step()"]
    G10 --> G11
    G11 --> G12["记录 transition"]
    G12 --> G2

    G2 --> REWARD["计算奖励"]
    REWARD --> R1["game.get_round_result()"]
    R1 --> R2["compute_team_rewards(result)"]
    R2 --> R3["每个玩家: +1(胜) / -1(负) / 0(半洞)"]

    R3 --> BUFFER["存入 replay buffer"]
    BUFFER --> TRAIN{"buffer 足够?"}
    TRAIN -->|否| LOOP
    TRAIN -->|是| T1["从 buffer 抽样 batch"]
    T1 --> T2["model.forward(state, pos, actions, lstm)"]
    T2 --> T3["q_values_max = q_values.max(dim=-1)[0]"]
    T3 --> T4["loss = F.mse_loss(q_values_max, rewards)"]
    T4 --> T5["反向传播 + optimizer.step()"]
    T5 --> LOG["记录日志"]

    LOG --> SAVE{"每 save_interval 步?"}
    SAVE -->|是| S1["保存 checkpoint"]
    SAVE -->|否| LOOP
    S1 --> LOOP
```

### 7.2 DMC损失函数

```mermaid
flowchart TD
    subgraph DMC_LOSS["DMC损失函数"]
        DL1["对每个transition:"] --> DL2["q_pred = model(state, position, actions, lstm)"]
        DL2 --> DL3["q_max = max(q_pred, dim=-1)"]
        DL3 --> DL4["target = team_reward (终局奖励)"]
        DL4 --> DL5["loss = MSE(q_max, target)"]
        DL5 --> DL6["注意: 无bootstrapping, 无TD, 纯MC"]
    end
```

### 7.3 DMC超参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `lr` | 1e-4 | 学习率 |
| `batch_size` | 256 | mini-batch大小 |
| `replay_buffer_size` | 100000 | 回放缓冲区大小 |
| `epsilon_start` | 1.0 | 初始探索率 |
| `epsilon_end` | 0.05 | 最终探索率 |
| `epsilon_decay` | 0.9999 | 探索率衰减 |
| `min_buffer_for_train` | 5000 | 开始训练的最小样本数 |

---

## 8. 奖励系统详解

### 8.1 PPO即时奖励

```mermaid
flowchart TD
    subgraph PPO_REWARD["PPO奖励函数 compute_step_reward()"]
        PR1["基础: 0"] --> PR2{"出牌使手牌减少?"}
        PR2 -->|是| PR3["+0.01 * (减少比例)"]
        PR2 -->|否| PR4["pass: -0.05"]
        PR3 --> PR5{"使用大牌?"}
        PR5 -->|是| PR6["-0.02 (炸弹扣分)"]
        PR5 -->|否| PR7["继续"]
        PR6 --> PR7
        PR7 --> PR8{"出完牌?"}
        PR8 -->|是| PR9["+0.5 (完成奖励)"]
        PR8 -->|否| PR10["继续"]
        PR9 --> PR10
        PR10 --> PR11{"队友出完?"}
        PR11 -->|是| PR12["+0.3 (团队协作)"]
        PR11 -->|否| PR13["继续"]
        PR12 --> PR13
        PR13 --> PR14{"对手出完?"}
        PR14 -->|是| PR15["-0.3 (对手完成惩罚)"]
        PR14 -->|否| PR16["返回 reward"]
        PR15 --> PR16
    end
```

### 8.2 DMC终局团队奖励

```mermaid
flowchart TD
    subgraph DMC_REWARD["DMC奖励函数 compute_team_rewards()"]
        DR1["game.get_round_result()"] --> DR2["获取 team_results"]
        DR2 --> DR3["对每个玩家:"]
        DR3 --> DR4{"所属队伍结果?"}
        DR4 -->|quan_dong| DR5["reward = +1.0"]
        DR4 -->|ban_dong| DR6["reward = +0.5"]
        DR4 -->|lose| DR7["reward = -1.0"]
        DR5 --> DR8["返回 {player: reward}"]
        DR6 --> DR8
        DR7 --> DR8
    end
```

### 8.3 奖励设计对比

| 维度 | PPO | DMC |
|------|-----|-----|
| 奖励类型 | 稠密即时奖励 | 稀疏终局奖励 |
| 信号频率 | 每步 | 每局结束 |
| 奖励范围 | [-0.3, 0.5+] | [-1, +1] |
| 团队信号 | 包含(队友/对手) | 包含(队伍结果) |
| 中间信号 | 手牌减少、大牌惩罚 | 无 |
| 信用分配 | 即时+GAE | 仅终局(无中间分配) |

---

## 9. 自对弈与模型共享

### 9.1 PPO自对弈架构

```mermaid
flowchart TD
    subgraph ENV["PPOEnvironment"]
        AI0["AI-Player 0<br/>模型实例 (共享)"]
        AI1["AI-Player 1<br/>模型实例 (共享)"]
        AI2["AI-Player 2<br/>模型实例 (共享)"]
        AI3["AI-Player 3<br/>模型实例 (共享)"]
    end

    subgraph MODEL["CardPolicyNetwork (单例)"]
        M["共享权重"]
    end

    AI0 -->|引用| M
    AI1 -->|引用| M
    AI2 -->|引用| M
    AI3 -->|引用| M

    subgraph COLLECT["采集数据"]
        C["4个玩家的经验都存入<br/>同一个 Replay Buffer"]
    end

    M -->|更新| C
    C -->|训练| M
```

### 9.2 模型checkpoint管理

```mermaid
flowchart TD
    SAVE["保存checkpoint"] --> S1["torch.save({model_state_dict, optimizer_state_dict, episode, ...})"]
    S1 --> S2["文件命名: ppo_ep{N}_{timestamp}.pt 或 douzero_ep{N}.pt"]
    S2 --> S3["default_rl.pt ← 最新模型符号链接"]

    LOAD["加载checkpoint"] --> L1["torch.load(filepath)"]
    L1 --> L2["model.load_state_dict(ckpt['model_state_dict'])"]
    L2 --> L3["optimizer.load_state_dict(ckpt['optimizer_state_dict'])"]
    L3 --> L4["episode = ckpt['episode']"]
```

---

## 10. 模型部署与推理

### 10.1 NeuralPolicy推理流程

```mermaid
flowchart TD
    ENTRY["_choose_play_with_model(last_ht, is_free)"] --> C1{"model_name in ('rule', 'heuristic', '')?"}
    C1 -->|是| C1A["直接调用规则AI:<br/>is_free? _find_free_play()<br/>: _find_beat_play(last_ht)"]
    C1 -->|否| C2["创建 fallback() 闭包<br/>(规则AI作为回退)"]
    C2 --> C3["NeuralPolicy(model_name).choose_play(...)"]
    C3 --> C4["get_shared_model(model_name)"]
    C4 --> C5{"模型加载成功?"}
    C5 -->|否| C6["返回 fallback_fn()"]
    C5 -->|是| C7["enumerate_legal_actions(hand, level_rank, last_ht, is_free_play)"]
    C7 --> C8{"有合法动作?"}
    C8 -->|否| C6
    C8 -->|是| C9["encode_state() → [160]"]
    C9 --> C10["encode_action() × N → [N, 48]"]
    C10 --> C11["encode_history() → [40, 64]"]
    C11 --> C12["torch.tensor() 转换张量"]
    C12 --> C13["with torch.no_grad():<br/>model(state, action, history)"]
    C13 --> C14["scores [N]"]
    C14 --> C15["argmax(scores) → best_idx"]
    C15 --> C16{"best_action['type'] == 'pass'?"}
    C16 -->|是| C17["返回 None (pass)"]
    C16 -->|否| C18["返回 best_action['indices']"]
    C13 -->|异常| C6
```

### 10.2 model_store单例缓存

```mermaid
flowchart TD
    GET["get_shared_model(model_name='default_rl')"] --> P1["resolve_model_path(model_name)"]
    P1 --> P2{"model_name 为空/rule/heuristic?"}
    P2 -->|是| P3["返回 None (不加载模型)"]
    P2 -->|否| P4{"model_name 是绝对路径?"}
    P4 -->|是| P5["直接使用该路径"]
    P4 -->|否| P6{"以 .pt/.pth 结尾?"}
    P6 -->|是| P7["join(checkpoints_dir, basename)"]
    P6 -->|否| P8["join(checkpoints_dir, model_name+.pt)"]
    P5 --> P9["cache_key = os.path.abspath(model_path)"]
    P7 --> P9
    P8 --> P9
    P9 --> P10["with _cache_lock:"]
    P10 --> P11{"cache_key in _model_cache?"}
    P11 -->|是| P12["返回缓存模型 (单例)"]
    P11 -->|否| P13["model = CardPolicyNetwork()"]
    P13 --> P14{"权重文件存在?"}
    P14 -->|否| P15["eval()后存入缓存, 返回"]
    P14 -->|是| P16["load_state_dict (兼容部分加载)"]
    P16 --> P15
```

---

## 11. 函数清单

### 11.1 rl/model.py

| 函数 | 签名 | 职责 | 输入/输出维度 |
|------|------|------|-------------|
| `CardPolicyNetwork.__init__` | `def __init__(self, state_dim=160, action_dim=48, history_dim=64, history_len=40, hidden_dim=256, nhead=4, num_transformer_layers=2)` | 构建完整Actor-Critic网络 | - |
| `forward` | `def forward(self, state_vec, action_vecs, history_vecs=None)` | 兼容旧推理接口：返回logits | ([B,160],[B,N,48],[B,40,64]) → [B,N] |
| `forward_actor_critic` | `def forward_actor_critic(self, state_vec, action_vecs, history_vecs)` | 前向传播：计算logits和value | ([B,160],[B,N,48],[B,40,64]) → ([B,N],[B,1]) |
| `forward_actor` | `def forward_actor(self, state_vec, action_vecs, history_vecs)` | 仅策略网络前向 | ([B,160],[B,N,48],[B,40,64]) → [B,N] |
| `forward_critic` | `def forward_critic(self, state_vec)` | 仅价值网络前向 | [B,160] → [B,1] |
| `get_action` | `def get_action(self, state_vec, action_vecs, history_vecs, deterministic=False)` | 获取动作（采样/贪心） | 输入同上 → (action_idx, log_prob, value) |
| `evaluate_actions` | `def evaluate_actions(self, state_vec, action_vecs, history_vecs, action_indices)` | 评估特定动作的log_prob和熵 | 输入同上+索引 → (log_probs, values, entropy) |

### 11.2 douzero_4a4/model.py

| 函数 | 签名 | 职责 | 输入/输出维度 |
|------|------|------|-------------|
| `DouZero4A4Model.__init__` | `def __init__(self, state_dim, action_dim, hidden_dim=256, lstm_hidden=512)` | 构建4个LSTM+Q头网络 | - |
| `forward` | `def forward(self, obs, position, action_features, lstm_state=None)` | 前向传播：计算Q值 | ([B,S],[B,1],[B,N,A],lstm) → ([B,N], lstm_state) |
| `get_lstm_state` | `def get_lstm_state(self, position)` | 获取指定位置的LSTM状态 | position → (h, c) |
| `set_lstm_state` | `def set_lstm_state(self, position, state)` | 设置指定位置的LSTM状态 | position, (h,c) → None |
| `reset_lstm_state` | `def reset_lstm_state(self, position=None)` | 重置LSTM状态 | position(可选) → None |

### 11.3 rl/policy.py

| 函数 | 签名 | 职责 |
|------|------|------|
| `NeuralPolicy.__init__` | `def __init__(self, model_name='default_rl')` | 初始化策略：通过 model_store 获取共用模型单例 |
| `choose_play` | `def choose_play(self, state, hand, level_rank, seat, last_ht, is_free_play, fallback_fn)` | 决策出牌：枚举合法动作→编码→模型推理→argmax→返回牌索引；失败时调用fallback_fn回退规则AI |

### 11.4 rl/model_store.py

| 函数 | 签名 | 职责 |
|------|------|------|
| `get_shared_model` | `def get_shared_model(model_name='default_rl')` | 获取共享模型实例：线程安全(_model_cache + _cache_lock)，同路径只加载一次 |
| `resolve_model_path` | `def resolve_model_path(model_name=None)` | 解析模型名称到权重文件路径；空名称/rule/heuristic 返回None（不加载模型） |
| `clear_model_cache` | `def clear_model_cache()` | 手动清空模型缓存（训练/调试时使用） |
| `cached_model_count` | `def cached_model_count()` | 返回当前进程内已加载模型数量 |

### 11.5 scripts/train_rl_ai.py

| 函数 | 签名 | 职责 |
|------|------|------|
| `main` | `def main()` | PPO训练入口 |
| `train` | `def train()` | PPO训练主循环 |
| `collect_rollouts` | `def collect_rollouts(env, model, buffer, horizon)` | 采集rollout数据 |
| `compute_gae` | `def compute_gae(rewards, values, dones, gamma, lam)` | 计算GAE优势 |
| `update_policy` | `def update_policy(model, optimizer, buffer, ppo_epochs, batch_size, clip_eps, vf_coef, ent_coef)` | PPO策略更新 |
| `save_checkpoint` | `def save_checkpoint(model, optimizer, episode, path)` | 保存checkpoint |
| `load_checkpoint` | `def load_checkpoint(path, model, optimizer)` | 加载checkpoint |

### 11.6 scripts/train_douzero_4a4.py

| 函数 | 签名 | 职责 |
|------|------|------|
| `main` | `def main()` | DMC训练入口 |
| `train` | `def train()` | DMC训练主循环 |
| `generate_episode` | `def generate_episode(model, env, epsilon)` | 生成一个完整回合 |
| `compute_rewards` | `def compute_rewards(game_result)` | 计算终局团队奖励 |
| `update_model` | `def update_model(model, optimizer, replay_buffer, batch_size)` | 模型更新 |
| `save_checkpoint` | `def save_checkpoint(model, optimizer, episode, path)` | 保存checkpoint |
| `load_checkpoint` | `def load_checkpoint(path, model, optimizer)` | 加载checkpoint |

### 11.7 douzero_4a4/rewards.py

| 函数 | 签名 | 职责 |
|------|------|------|
| `compute_team_rewards` | `def compute_team_rewards(round_result)` | 根据终局结果计算每个玩家的奖励 |
| `get_reward_for_player` | `def get_reward_for_player(player_seat, team_results)` | 获取单个玩家的奖励 |

---

## 附录A: 训练日志示例

### A.1 健康的PPO训练日志

```
[Episode 1000] avg_reward: -0.12 | policy_loss: 0.0234 | value_loss: 0.1567 | entropy: 2.34 | avg_ep_len: 45.2
[Episode 2000] avg_reward: -0.05 | policy_loss: 0.0189 | value_loss: 0.1321 | entropy: 2.21 | avg_ep_len: 42.8
[Episode 5000] avg_reward: 0.03  | policy_loss: 0.0142 | value_loss: 0.0987 | entropy: 2.05 | avg_ep_len: 38.1
[Episode 10000] avg_reward: 0.15 | policy_loss: 0.0098 | value_loss: 0.0723 | entropy: 1.89 | avg_ep_len: 34.5
[Episode 20000] avg_reward: 0.28 | policy_loss: 0.0067 | value_loss: 0.0512 | entropy: 1.72 | avg_ep_len: 30.2
```

### A.2 健康的DMC训练日志

```
[Episode 1000] win_rate: 0.48 | avg_q: 0.12 | loss: 0.5234 | epsilon: 0.95 | buffer: 5230
[Episode 5000] win_rate: 0.52 | avg_q: 0.23 | loss: 0.4123 | epsilon: 0.78 | buffer: 25000
[Episode 10000] win_rate: 0.56 | avg_q: 0.38 | loss: 0.3215 | epsilon: 0.45 | buffer: 50000
[Episode 20000] win_rate: 0.62 | avg_q: 0.51 | loss: 0.2456 | epsilon: 0.12 | buffer: 100000
[Episode 50000] win_rate: 0.71 | avg_q: 0.67 | loss: 0.1876 | epsilon: 0.05 | buffer: 100000
```

### A.3 异常训练日志警示

```
[Episode 5000] avg_reward: -0.45 | policy_loss: 0.0012 | value_loss: 0.4521 | entropy: 0.12 | avg_ep_len: 12.3
  ↑ 熵太低 → 策略已崩溃，过早收敛
[Episode 10000] avg_reward: -0.50 | policy_loss: 0.0001 | value_loss: 0.8912 | entropy: 0.01 | avg_ep_len: 8.1
  ↑ 完全崩溃，需要增大ent_coef
[Episode 5000] win_rate: 0.33 | avg_q: 0.89 | loss: 0.0234 | epsilon: 0.05 | buffer: 5000
  ↑ Q值过高 → 过估计问题，buffer太小
[Episode 20000] win_rate: 0.50 | avg_q: 0.01 | loss: 0.0012 | epsilon: 0.05 | buffer: 100000
  ↑ Q值趋近0 → 梯度消失，学习率太低或模型容量不足
```

---

## 附录B: 文件索引

| 文件 | 路径 | 职责 |
|------|------|------|
| `rl/model.py` | `d:\workspace\4A4\rl\model.py` | CardPolicyNetwork定义 |
| `rl/features.py` | `d:\workspace\4A4\rl\features.py` | 特征编码 |
| `rl/actions.py` | `d:\workspace\4A4\rl\actions.py` | 合法动作枚举 |
| `rl/policy.py` | `d:\workspace\4A4\rl\policy.py` | NeuralPolicy决策器 |
| `rl/model_store.py` | `d:\workspace\4A4\rl\model_store.py` | 模型单例缓存 |
| `douzero_4a4/model.py` | `d:\workspace\4A4\douzero_4a4\model.py` | DouZero4A4Model |
| `douzero_4a4/rewards.py` | `d:\workspace\4A4\douzero_4a4\rewards.py` | 奖励函数 |
| `scripts/train_rl_ai.py` | `d:\workspace\4A4\scripts\train_rl_ai.py` | PPO训练 |
| `scripts/train_douzero_4a4.py` | `d:\workspace\4A4\scripts\train_douzero_4a4.py` | DMC训练 |