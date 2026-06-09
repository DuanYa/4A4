# 4A4 微信横屏小游戏

`minigame` 目录已从微信官方初始模板改造成 4A4 横屏 Canvas 客户端。入口仍是 `game.js`，核心逻辑在 `js/main.js`。

## 功能

- 横屏牌桌、房间等待、座位展示、手牌选择、出牌、不出、叉牌、点牌、结算。
- 使用后端现有接口：`POST /api/rooms` 创建房间，Socket.IO 事件加入房间和同步牌局。
- 不依赖 DOM、Phaser 或浏览器版 socket.io；通过 `wx.request` 和 `wx.connectSocket` 直接连接后端。
- 启动时调用 `wx.login`，后端使用 SQLite 保存微信用户、房间成员、房间快照和动作流水。
- `wx.login` 是静默登录，不会弹出授权框；需要展示微信头像昵称时，点击首屏或房间内的“授权头像”按钮。
- 关闭小游戏再打开时，会优先恢复上次房间或正在进行的对局；从分享进入时优先加入分享房间。
- 房间和牌局内可通过微信小游戏分享机制邀请好友，分享参数为 `room_id`。

## 使用

1. 用微信开发者工具打开 `minigame` 目录。
2. 确认 `game.json` 中 `deviceOrientation` 为 `landscape`。
3. 开发阶段可关闭“校验合法域名、web-view 域名、TLS 版本以及 HTTPS 证书”。
4. 真机调试时，需要在微信后台配置后端域名的 request 合法域名和 socket 合法域名。
5. 首屏点按“服务器”“房间”“昵称”可用微信键盘修改配置；房间号留空时会自动创建。

## 微信登录与 SQLite

后端默认使用 SQLite，数据库文件位于：

```text
data/4a4.sqlite3
```

可通过环境变量修改位置：

```text
FOURA4_DB_PATH=/path/to/4a4.sqlite3
```

正式微信登录需要在后端设置：

```text
WECHAT_APPID=你的appid
WECHAT_SECRET=你的secret
```

开发者工具或本地没有密钥时，后端会使用小游戏本地保存的 `anonymous_id` 创建稳定开发账号，便于测试恢复房间和分享进入流程。

默认服务器地址：

```text
http://dyeea.com:5000
```

客户端会自动把 HTTP/HTTPS 地址转换为 Socket.IO WebSocket 地址：

```text
ws://dyeea.com:5000/socket.io/?EIO=4&transport=websocket
```
