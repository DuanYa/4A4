# 四幺四(4A4)扑克游戏 - 设计文档

## 一、项目概述

四幺四是一个前后端分离的在线多人扑克游戏，支持4人实时对战。

**技术栈：**
- 后端：Python 3 + Flask + Flask-SocketIO
- 前端：原生 HTML/CSS/JavaScript + Socket.IO Client
- 通信：REST API（房间管理）+ WebSocket（游戏实时通信）
- 并发：多线程（threading模式）

## 二、架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────┐
│              前端 (Browser)              │
│  ┌──────┐  ┌──────┐  ┌──────┐           │
│  │app.js│  │game.js│  │ ws.js│           │
│  └──┬───┘  └──┬───┘  └──┬───┘           │
│     └─────────┴─────────┘               │
│              Socket.IO / HTTP            │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────┴──────────────────────┐
│           后端 (Flask + SocketIO)         │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │  routes/api   │  │   routes/ws     │  │
│  │  (REST API)   │  │  (WebSocket)    │  │
│  └──────┬───────┘  └────────┬────────┘  │
│         └────────┬──────────┘            │
│                  │                       │
│  ┌───────────────┴──────────────────┐   │
│  │         models/ (游戏规则层)       │   │
│  │  card.py   hand_type.py  game.py │   │
│  │  deck.py   player.py    room.py  │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### 2.2 分层设计原则

| 层 | 职责 | 依赖 |
|----|------|------|
| models | 纯游戏规则逻辑 | 无外部依赖 |
| routes | 路由/通信/会话管理 | Flask, models |
| static | UI渲染与交互 | Socket.IO |

**关键设计：模型层与后端完全隔离。** `models/` 下的代码不依赖 Flask 或任何网络库，可以独立测试和复用。

## 三、数据模型

### 3.1 Card（牌）
- `suit`: 花色枚举（SPADE/HEART/CLUB/DIAMOND/JOKER）
- `rank`: 点数字符串（3-A, 2, BJ, RJ）
- `get_value(level_rank)`: 返回带级牌加成的权重值

### 3.2 HandType（牌型）
```
单张 < 对子 < 单龙 < 双龙 < 炸 < 轰 < 双王 < 四幺四
```

8种牌型的识别与比较逻辑在 `hand_type.py` 中实现。

### 3.3 Player（玩家）
- 手牌管理、座位/队伍信息
- 叉/点相关的手牌查询方法

### 3.4 Game（单局游戏）
- 完整的出牌状态机（见下文）
- 线程安全（threading.Lock）

### 3.5 Room（房间）
- 玩家管理、升级系统
- 多局游戏的生命周期管理

## 四、时序设计（重点）

### 4.1 游戏状态机

```
                    ┌──────────┐
                    │ WAITING  │
                    └────┬─────┘
                         │ start()
                    ┌────▼─────┐
                    │ DEALING  │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
               ┌───>│ PLAYING  │<────────────────┐
               │    └────┬─────┘                  │
               │         │                        │
               │    出单张(非王)                    │
               │         │                        │
               │    ┌────▼──────┐   不叉/全不叉    │
               │    │CHA_ASKING ├────────────────>│
               │    └────┬──────┘                  │
               │         │ 叉牌                    │
               │    ┌────▼──────┐   不点/全不点     │
               │    │DIAN_ASKING├────────────────>│
               │    └────┬──────┘                  │
               │         │ 点牌                    │
               │         └───────────────────────>│
               │                                   │
               │    3人出完                         │
               │    ┌────▼─────┐                   │
               └────│ROUND_END │                   │
                    └──────────┘
```

### 4.2 叉/点跳跃式牌权时序

叉和点会打断正常的顺时针出牌顺序：

```
正常流程:  P0 → P1 → P2 → P3 → P0 → ...

叉牌场景:  P0出单张5 → 询问P1(有对5?) → 询问P2 → P2叉！
           → 询问P3(有5?) → P3点！→ P3获得出牌权
           → P3 → P0 → P1 → P2 → ...  (恢复正常)
```

**询问流程：**
1. 出单张后，顺时针依次询问每个持有对子的玩家是否叉
2. 叉牌后，顺时针依次询问每个持有单张的玩家是否点
3. 叉/点的响应通过WebSocket异步完成

### 4.3 接风规则

```
P0出完牌 → 其余人都pass → P2(P0队友)接风获得出牌权
P0出完牌 且 P2也出完 → 牌权给下一个活跃玩家
```

## 五、通信协议

### 5.1 REST API

| 端点 | 方法 | 说明 |
|------|------|------|
| /api/rooms | GET | 获取房间列表 |
| /api/rooms | POST | 创建房间 |
| /api/rooms/:id | GET | 获取房间信息 |

### 5.2 WebSocket 事件

**客户端 → 服务端：**

| 事件 | 数据 | 说明 |
|------|------|------|
| join_room | {room_id, name} | 加入房间 |
| start_game | {} | 开始游戏 |
| play_cards | {card_indices} | 出牌 |
| pass_turn | {} | 不出 |
| respond_cha | {do_cha} | 回应叉牌 |
| respond_dian | {do_dian} | 回应点牌 |

**服务端 → 客户端：**

| 事件 | 说明 |
|------|------|
| joined | 加入成功 |
| room_state | 房间状态更新 |
| game_state | 游戏状态（按玩家视角） |
| game_action | 游戏动作广播 |
| round_end | 一局结束结算 |
| error | 错误信息 |

### 5.3 信息隐藏

`game_state` 事件为每个玩家生成独立视角：
- 只能看到自己的手牌
- 其他玩家只显示手牌数量

## 六、线程安全

- `Game.lock`: 保护单局游戏的所有状态修改
- `session_lock`: 保护玩家会话映射
- `Room.lock`: 保护房间玩家列表修改
- Flask-SocketIO 配置 `async_mode='threading'`

## 七、升级系统

```
级别: 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → J → Q → K → A

全洞(1,2名同队): 升2级
半洞(1,3名同队): 升1级
打3/J/A: 必须全洞才能升

直J(打J被反洞): 退回打3
直A(打A被反洞): 退回打J
```

## 八、文件结构

```
cursor-4a4/
├── server.py              # 服务入口
├── requirements.txt       # 依赖
├── DESIGN.md              # 本文档
├── GAME_RULES.md          # 游戏规则
├── models/
│   ├── __init__.py
│   ├── card.py            # 牌/花色
│   ├── deck.py            # 洗牌/发牌
│   ├── hand_type.py       # 牌型识别/比较
│   ├── player.py          # 玩家
│   ├── game.py            # 单局游戏逻辑
│   ├── game_actions.py    # 叉/点/pass逻辑
│   ├── room.py            # 房间/升级系统
│   ├── ai_player.py       # AI玩家WebSocket客户端
│   └── ai_search.py       # AI出牌搜索算法
├── routes/
│   ├── __init__.py
│   ├── api.py             # REST路由
│   └── ws.py              # WebSocket处理
└── static/
    ├── index.html
    ├── css/style.css
    └── js/
        ├── ws.js           # WebSocket客户端
        ├── game.js         # 游戏渲染
        └── app.js          # 应用入口

## 九、AI玩家系统

### 9.1 架构设计

AI玩家作为独立的Python SocketIO客户端连接服务器，对服务器而言与真实玩家无异。

`
服务器 (Flask + SocketIO)
    │
    ├── WebSocket ── 浏览器玩家 (前端JS)
    ├── WebSocket ── AI玩家 (Python socketio.Client)
    ├── WebSocket ── AI玩家 (Python socketio.Client)
    └── WebSocket ── AI玩家 (Python socketio.Client)
`

### 9.2 AI补位流程

`
玩家点击“开始游戏”
    │
    ├─ 房间已满4人 → 直接开始
    └─ 房间未满 → 后端自动创建AI玩家补位
        │
        ├─ 创建AIPlayer实例
        ├─ 通过socketio.Client连接服务器
        ├─ 加入房间
        └─ 全部就位后开始游戏
`

### 9.3 AI决策逻辑

| 场景 | AI行为 |
|------|--------|
| 自由出牌 | 出最小的单张 |
| 跟牌 | 找能管住上家的最小组合，找不到则pass |
| 叉牌询问 | 总是叉 |
| 点牌询问 | 总是点 |

AI搜索顺序：同类型跟牌 → 炸弹（炸/轰/双王/四幺四）

### 9.4 叉/点时序适配

AI通过监听 game_state 事件判断当前状态：
- cha_asking_seat == 自己的座位 → 响应叉牌
- dian_asking_seat == 自己的座位 → 响应点牌
- current_player_seat == 自己的座位 → 出牌/pass

所有AI决策延迟0.5秒执行，模拟思考时间。

## 十、运行方式

```bash
pip install -r requirements.txt
python server.py
# 访问 http://localhost:5000
```

打开4个浏览器窗口，加入同一房间即可游戏。
