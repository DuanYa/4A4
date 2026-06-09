const BASE_WIDTH = 1334;
const BASE_HEIGHT = 750;
const DEFAULT_SERVER = 'http://dyeea.com:5000';
const STORAGE_KEY = '4a4_minigame_config';

const gameCanvas = typeof canvas !== 'undefined' ? canvas : wx.createCanvas();
const ctx = gameCanvas.getContext('2d');

function now() {
  return Date.now();
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function padRoomId() {
  return String(Math.floor(Math.random() * 900000 + 100000));
}

function randomId(prefix) {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 12)}`;
}

function normalizeServerUrl(value) {
  const raw = String(value || '').trim() || DEFAULT_SERVER;
  return raw.replace(/\/+$/, '');
}

function isLoopbackServer(value) {
  return /^https?:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/i.test(String(value || '').replace(/\/+$/, ''));
}

function socketUrlFromServer(serverUrl) {
  const base = normalizeServerUrl(serverUrl);
  const wsBase = base.replace(/^https:/i, 'wss:').replace(/^http:/i, 'ws:');
  return `${wsBase}/socket.io/?EIO=4&transport=websocket`;
}

function safeJsonParse(text, fallback) {
  try {
    return JSON.parse(text);
  } catch (e) {
    return fallback;
  }
}

function normalizeErrorMessage(data) {
  const code = data && (data.code || data.message);
  if (code === 'cannot_beat') return '管不上，请换牌或选择不出';
  return (data && data.message) || '操作失败';
}

function cardRank(card) {
  return card && card.rank ? String(card.rank) : '';
}

function cardSuit(card) {
  return card && card.suit ? String(card.suit) : '';
}

function suitSymbol(suit) {
  return {
    spade: '♠',
    heart: '♥',
    club: '♣',
    diamond: '♦',
    joker: ''
  }[suit] || '';
}

function isRedCard(card) {
  const suit = cardSuit(card);
  const rank = cardRank(card);
  return suit === 'heart' || suit === 'diamond' || rank === 'RJ';
}

function cardText(card) {
  if (!card) return '';
  const rank = cardRank(card);
  if (rank === 'RJ') return '大王';
  if (rank === 'BJ') return '小王';
  return `${suitSymbol(cardSuit(card))}${rank}`;
}

function shortCards(cards, limit = 10) {
  if (!cards || !cards.length) return '';
  const list = cards.slice(0, limit).map(cardText);
  if (cards.length > limit) list.push(`+${cards.length - limit}`);
  return list.join(' ');
}

function roundRect(g, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  g.beginPath();
  g.moveTo(x + rr, y);
  g.lineTo(x + w - rr, y);
  g.quadraticCurveTo(x + w, y, x + w, y + rr);
  g.lineTo(x + w, y + h - rr);
  g.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
  g.lineTo(x + rr, y + h);
  g.quadraticCurveTo(x, y + h, x, y + h - rr);
  g.lineTo(x, y + rr);
  g.quadraticCurveTo(x, y, x + rr, y);
  g.closePath();
}

function drawText(g, text, x, y, size, color, align = 'left', weight = 'normal') {
  g.font = `${weight} ${size}px Arial, "Microsoft YaHei", sans-serif`;
  g.fillStyle = color;
  g.textAlign = align;
  g.textBaseline = 'middle';
  g.fillText(String(text || ''), x, y);
}

function drawFittedText(g, text, x, y, maxWidth, size, color, align = 'center', weight = 'normal') {
  let fontSize = size;
  const content = String(text || '');
  while (fontSize > 14) {
    g.font = `${weight} ${fontSize}px Arial, "Microsoft YaHei", sans-serif`;
    if (g.measureText(content).width <= maxWidth) break;
    fontSize -= 1;
  }
  g.fillStyle = color;
  g.textAlign = align;
  g.textBaseline = 'middle';
  g.fillText(content, x, y);
}

function loadConfig() {
  const saved = safeJsonParse(wx.getStorageSync(STORAGE_KEY) || '{}', {});
  const savedServer = normalizeServerUrl(saved.serverUrl || DEFAULT_SERVER);
  return {
    serverUrl: isLoopbackServer(savedServer) ? DEFAULT_SERVER : savedServer,
    roomId: String(saved.roomId || ''),
    name: String(saved.name || `玩家${Math.floor(Math.random() * 900 + 100)}`),
    avatarUrl: String(saved.avatarUrl || ''),
    anonymousId: String(saved.anonymousId || randomId('wxdev')),
    sessionToken: String(saved.sessionToken || ''),
    userId: saved.userId || null
  };
}

function saveConfig(config) {
  wx.setStorageSync(STORAGE_KEY, JSON.stringify({
    serverUrl: normalizeServerUrl(config.serverUrl),
    roomId: String(config.roomId || ''),
    name: String(config.name || ''),
    avatarUrl: String(config.avatarUrl || ''),
    anonymousId: String(config.anonymousId || ''),
    sessionToken: String(config.sessionToken || ''),
    userId: config.userId || null
  }));
}

class MiniSocket {
  constructor() {
    this.url = '';
    this.task = null;
    this.connected = false;
    this.opened = false;
    this.listeners = {};
    this.queue = [];
  }

  on(event, handler) {
    if (!this.listeners[event]) this.listeners[event] = [];
    this.listeners[event].push(handler);
  }

  off(event, handler) {
    const list = this.listeners[event];
    if (!list) return;
    this.listeners[event] = list.filter((item) => item !== handler);
  }

  fire(event, payload) {
    (this.listeners[event] || []).forEach((handler) => handler(payload));
  }

  connect(serverUrl) {
    const nextUrl = socketUrlFromServer(serverUrl);
    if (this.task && this.url === nextUrl && (this.opened || this.connected)) {
      return;
    }
    this.close();
    this.url = nextUrl;
    this.connected = false;
    this.opened = false;
    this.task = wx.connectSocket({ url: nextUrl });
    this.task.onOpen(() => {
      this.opened = true;
      this.fire('transport_open');
    });
    this.task.onMessage((message) => this.handleMessage(message.data));
    this.task.onError((error) => {
      this.fire('error', { message: 'WebSocket 连接失败', detail: error });
    });
    this.task.onClose(() => {
      this.opened = false;
      this.connected = false;
      this.fire('disconnect');
    });
  }

  close() {
    if (!this.task) return;
    try {
      this.task.close();
    } catch (e) {}
    this.task = null;
    this.connected = false;
    this.opened = false;
  }

  sendPacket(packet) {
    if (!this.task || !this.opened) return;
    this.task.send({ data: packet });
  }

  emit(event, data) {
    const payload = `42${JSON.stringify([event, data || {}])}`;
    if (!this.connected) {
      this.queue.push(payload);
      return;
    }
    this.sendPacket(payload);
  }

  flushQueue() {
    const pending = this.queue.splice(0);
    pending.forEach((packet) => this.sendPacket(packet));
  }

  handleMessage(rawData) {
    const packet = typeof rawData === 'string' ? rawData : String(rawData || '');
    if (!packet) return;
    const type = packet[0];
    if (type === '0') {
      this.sendPacket('40');
      return;
    }
    if (packet === '2') {
      this.sendPacket('3');
      return;
    }
    if (packet.indexOf('40') === 0) {
      this.connected = true;
      this.fire('connected');
      this.flushQueue();
      return;
    }
    if (packet.indexOf('42') === 0) {
      const args = safeJsonParse(packet.slice(2), []);
      if (Array.isArray(args) && args.length) {
        this.fire(args[0], args[1]);
      }
    }
  }
}

class Button {
  constructor(id, text, x, y, w, h, onTap, tone = 'green') {
    this.id = id;
    this.text = text;
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.onTap = onTap;
    this.tone = tone;
    this.enabled = true;
    this.visible = true;
  }

  contains(x, y) {
    return this.visible && this.enabled
      && x >= this.x && x <= this.x + this.w
      && y >= this.y && y <= this.y + this.h;
  }

  draw(g) {
    if (!this.visible) return;
    const palette = {
      green: ['#26a269', '#63e6a4'],
      amber: ['#d59a00', '#ffe07a'],
      blue: ['#2475d6', '#8bc8ff'],
      gray: ['#44515b', '#aebbc2'],
      red: ['#a43b45', '#ff9aa2']
    }[this.tone] || ['#26a269', '#63e6a4'];
    g.save();
    g.globalAlpha = this.enabled ? 1 : 0.45;
    roundRect(g, this.x, this.y, this.w, this.h, 10);
    g.fillStyle = palette[0];
    g.fill();
    g.lineWidth = 2;
    g.strokeStyle = palette[1];
    g.stroke();
    drawFittedText(g, this.text, this.x + this.w / 2, this.y + this.h / 2 + 1, this.w - 22, 23, '#ffffff', 'center', 'bold');
    g.restore();
  }
}

export default class Main {
  constructor() {
    this.config = loadConfig();
    this.applyLaunchQuery(wx.getLaunchOptionsSync ? wx.getLaunchOptionsSync() : null);
    this.socket = new MiniSocket();
    this.screen = 'lobby';
    this.roomState = null;
    this.gameState = null;
    this.result = null;
    this.mySeat = -1;
    this.isHost = false;
    this.buttons = [];
    this.cardRects = [];
    this.imageCache = {};
    this.selected = new Set();
    this.touchSelectMode = true;
    this.logs = [];
    this.toast = null;
    this.editField = null;
    this.loginReady = !!this.config.sessionToken;
    this.profileReady = !!this.config.avatarUrl;
    this.resumeChecked = false;
    this.shareRoomId = this.config.roomId;
    this.profileButtonRect = null;
    this.nativeProfileButton = null;
    this.nativeProfileButtonKey = '';
    this.lastFrameTime = now();
    this.animationId = 0;
    this.dpr = 1;
    this.viewWidth = 0;
    this.viewHeight = 0;
    this.scale = 1;
    this.offsetX = 0;
    this.offsetY = 0;
    this.bindSocket();
    this.bindInput();
    this.setupShare();
    this.resize();
    this.loop();
    this.loginWithWechat();
  }

  bindSocket() {
    this.socket.on('connected', () => this.showToast('服务器已连接'));
    this.socket.on('disconnect', () => this.showToast('连接已断开', 2200));
    this.socket.on('error', (data) => {
      if (data && data.code === 'cannot_beat' && this.pendingPlayIndices) {
        this.selected = new Set(this.pendingPlayIndices);
      }
      this.showToast(normalizeErrorMessage(data), 2600);
    });
    this.socket.on('joined', (data) => {
      this.mySeat = data.seat;
      this.config.roomId = data.room_id || this.config.roomId;
      this.shareRoomId = this.config.roomId;
      this.roomState = data.room_state || null;
      this.screen = 'room';
      this.result = null;
      this.selected.clear();
      saveConfig(this.config);
      this.showToast(`已进入房间 ${this.config.roomId}`);
    });
    this.socket.on('seat_changed', (data) => {
      this.mySeat = data.seat;
      this.showToast(`已切到座位 ${data.seat}`);
    });
    this.socket.on('room_state', (data) => {
      this.roomState = data || this.roomState;
      if (this.screen === 'lobby') this.screen = 'room';
    });
    this.socket.on('game_state', (data) => {
      this.gameState = data;
      this.screen = 'game';
      this.syncHandSelection();
    });
    this.socket.on('game_action', (data) => {
      if (data && data.seat === this.mySeat && data.cards && data.cards.length) {
        this.pendingPlayIndices = null;
        this.selected.clear();
      }
      this.addActionLog(data);
    });
    this.socket.on('round_end', (data) => {
      this.result = data;
      this.screen = 'game';
    });
    this.socket.on('profile_updated', (data) => {
      if (data && data.room_state) this.roomState = data.room_state;
      this.showToast('头像已同步');
    });
    this.socket.on('player_left', (data) => {
      this.showToast(`座位 ${data.seat} 离开了房间`);
    });
  }

  bindInput() {
    wx.onTouchStart((event) => this.handleTouchStart(event));
    wx.onTouchMove((event) => this.handleTouchMove(event));
    wx.onTouchEnd(() => {
      this.draggingCards = false;
    });
    if (wx.onKeyboardConfirm) wx.onKeyboardConfirm((event) => this.commitKeyboard(event.value));
    if (wx.onKeyboardComplete) wx.onKeyboardComplete((event) => this.commitKeyboard(event.value));
    if (wx.onWindowResize) wx.onWindowResize(() => this.resize());
    if (wx.onShow) wx.onShow((options) => this.applyLaunchQuery(options));
  }

  applyLaunchQuery(options) {
    const query = options && options.query ? options.query : {};
    const roomId = query.room_id || query.roomId;
    if (!roomId) return;
    this.config.roomId = String(roomId);
    this.shareRoomId = this.config.roomId;
    saveConfig(this.config);
    if (this.screen === 'lobby') {
      this.showToast(`收到房间邀请 ${this.config.roomId}`, 2400);
    }
  }

  setupShare() {
    if (wx.showShareMenu) {
      wx.showShareMenu({ withShareTicket: true, menus: ['shareAppMessage'] });
    }
    if (wx.onShareAppMessage) {
      wx.onShareAppMessage(() => this.sharePayload());
    }
  }

  logicalRectToScreen(x, y, w, h) {
    return {
      left: this.offsetX + x * this.scale,
      top: this.offsetY + y * this.scale,
      width: w * this.scale,
      height: h * this.scale
    };
  }

  sharePayload() {
    const roomId = this.config.roomId || this.shareRoomId || '';
    return {
      title: roomId ? `邀请你加入4A4房间 ${roomId}` : '邀请你来玩4A4',
      query: roomId ? `room_id=${encodeURIComponent(roomId)}` : '',
    };
  }

  shareRoom() {
    if (!this.config.roomId) {
      this.showToast('还没有可分享的房间', 1800);
      return;
    }
    if (wx.shareAppMessage) {
      wx.shareAppMessage(this.sharePayload());
    } else {
      this.showToast('当前微信版本不支持主动分享', 2200);
    }
  }

  requestJson(path, method, data, onSuccess, onFail) {
    const primary = normalizeServerUrl(this.config.serverUrl);
    const servers = primary === DEFAULT_SERVER ? [primary] : [primary, DEFAULT_SERVER];
    const attempt = (index) => {
      const serverUrl = servers[index];
      wx.request({
        url: `${serverUrl}${path}`,
        method,
        header: { 'content-type': 'application/json' },
        data: data || {},
        success: (res) => {
          if (res.statusCode >= 400 || (res.data && res.data.error)) {
            if (index + 1 < servers.length) {
              attempt(index + 1);
              return;
            }
            if (onFail) onFail(res.data || { error: `HTTP ${res.statusCode}`, serverUrl });
            return;
          }
          if (serverUrl !== this.config.serverUrl) {
            this.config.serverUrl = serverUrl;
            saveConfig(this.config);
            this.showToast('已切换到远端服务器', 1600);
          }
          onSuccess(res.data || {});
        },
        fail: () => {
          if (index + 1 < servers.length) {
            attempt(index + 1);
            return;
          }
          if (onFail) onFail({ error: 'network_failed', serverUrl });
        }
      });
    };
    attempt(0);
  }

  loginWithWechat() {
    const doLogin = (code) => {
      this.requestJson('/api/wx/login', 'POST', {
        code,
        anonymous_id: this.config.anonymousId,
        nickname: this.config.name,
        avatar_url: this.config.avatarUrl
      }, (data) => {
        this.config.sessionToken = data.session_token || '';
        this.config.userId = data.user_id || null;
        if (!this.config.avatarUrl && data.avatar_url) this.config.avatarUrl = data.avatar_url;
        if (data.nickname && (!this.config.name || this.config.name.indexOf('玩家') === 0)) {
          this.config.name = data.nickname;
        }
        this.loginReady = !!this.config.sessionToken;
        this.profileReady = !!this.config.avatarUrl;
        saveConfig(this.config);
        this.showToast(this.loginReady ? '微信登录完成' : '登录返回异常', 1600);
        this.tryResumeSession();
      }, () => {
        this.loginReady = !!this.config.sessionToken;
        this.showToast('本地身份继续，稍后可重试登录', 2200);
      });
    };
    if (wx.login) {
      wx.login({
        success: (res) => doLogin(res.code || ''),
        fail: () => doLogin('')
      });
    } else {
      doLogin('');
    }
  }

  authorizeUserProfile() {
    if (wx.chooseAvatar) {
      wx.chooseAvatar({
        success: (res) => {
          if (res && res.avatarUrl) {
            this.applyUserProfile({ avatarUrl: res.avatarUrl });
            return;
          }
          this.showToast('未选择微信头像', 1800);
        },
        fail: () => this.showToast('未选择微信头像', 1800)
      });
      return;
    }
    if (wx.createUserInfoButton) {
      this.showToast('请直接点击授权头像按钮', 1800);
      this.syncNativeProfileButton();
      return;
    }
    if (wx.getUserProfile) {
      wx.getUserProfile({
        desc: '用于在4A4房间中展示微信头像和昵称',
        success: (res) => this.applyUserProfile(res.userInfo || {}),
        fail: () => this.showToast('未授权头像昵称，请在微信弹窗中允许', 2200)
      });
      return;
    }
    this.showToast('当前基础库不支持头像授权', 2200);
  }

  applyUserProfile(userInfo) {
    const nickName = userInfo.nickName || userInfo.nickname || '';
    const avatarUrl = userInfo.avatarUrl || userInfo.avatar_url || '';
    if (nickName) this.config.name = nickName;
    if (avatarUrl) this.config.avatarUrl = avatarUrl;
    this.profileReady = !!this.config.avatarUrl;
    saveConfig(this.config);
    this.syncProfileToServer();
    this.showToast(this.profileReady ? '微信头像已授权' : '昵称已授权', 1800);
  }

  tryGetUserInfoFallback() {
    if (!wx.getUserInfo) {
      this.showToast('当前微信不再返回头像昵称，可点昵称手动修改', 2400);
      return;
    }
    wx.getUserInfo({
      lang: 'zh_CN',
      withCredentials: false,
      success: (res) => {
        if (res && res.userInfo) {
          this.applyUserProfile(res.userInfo);
        } else {
          this.showToast('当前微信未返回头像昵称，可点昵称手动修改', 2400);
        }
      },
      fail: () => this.showToast('当前微信未返回头像昵称，可点昵称手动修改', 2400)
    });
  }

  syncProfileToServer() {
    if (!this.config.sessionToken) return;
    const payload = {
      session_token: this.config.sessionToken,
      nickname: this.config.name,
      name: this.config.name,
      avatar_url: this.config.avatarUrl
    };
    this.requestJson('/api/wx/profile', 'POST', payload, () => {}, () => {});
    if (this.socket && this.socket.connected && this.config.roomId) {
      this.socket.emit('update_profile', payload);
    }
  }

  tryResumeSession() {
    if (this.resumeChecked || !this.config.sessionToken) return;
    this.resumeChecked = true;
    this.requestJson('/api/session/resume', 'POST', {
      session_token: this.config.sessionToken
    }, (data) => {
      if (!data.has_session) return;
      if (this.config.roomId && this.config.roomId !== data.room_id) return;
      this.config.roomId = data.room_id;
      this.mySeat = data.seat;
      this.isHost = !!data.is_host;
      this.roomState = data.room_state || null;
      this.gameState = data.game_state || null;
      this.screen = this.gameState ? 'game' : 'room';
      saveConfig(this.config);
      this.joinRoom(this.isHost, { restoring: true });
      this.showToast(data.stale ? '已读取历史快照' : '已恢复上次房间', 2200);
    }, () => {});
  }

  resize() {
    const info = wx.getSystemInfoSync();
    this.dpr = info.pixelRatio || 1;
    this.viewWidth = info.windowWidth || BASE_WIDTH;
    this.viewHeight = info.windowHeight || BASE_HEIGHT;
    gameCanvas.width = Math.floor(this.viewWidth * this.dpr);
    gameCanvas.height = Math.floor(this.viewHeight * this.dpr);
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.scale = Math.min(this.viewWidth / BASE_WIDTH, this.viewHeight / BASE_HEIGHT);
    this.offsetX = (this.viewWidth - BASE_WIDTH * this.scale) / 2;
    this.offsetY = (this.viewHeight - BASE_HEIGHT * this.scale) / 2;
  }

  loop() {
    this.render();
    this.animationId = requestAnimationFrame(() => this.loop());
  }

  toLogical(touch) {
    return {
      x: (touch.clientX - this.offsetX) / this.scale,
      y: (touch.clientY - this.offsetY) / this.scale
    };
  }

  makeButton(id, text, x, y, w, h, onTap, tone) {
    const btn = new Button(id, text, x, y, w, h, onTap, tone);
    this.buttons.push(btn);
    return btn;
  }

  makeProfileButton(text, x, y, w, h, tone) {
    this.profileButtonRect = { text, x, y, w, h, tone };
    return this.makeButton('profile', text, x, y, w, h, () => this.authorizeUserProfile(), tone);
  }

  syncNativeProfileButton() {
    if (!wx.createUserInfoButton || !this.profileButtonRect) {
      this.destroyNativeProfileButton();
      return;
    }
    const r = this.profileButtonRect;
    const screen = this.logicalRectToScreen(r.x, r.y, r.w, r.h);
    const key = [
      this.screen, r.text,
      Math.round(screen.left), Math.round(screen.top),
      Math.round(screen.width), Math.round(screen.height)
    ].join(':');
    if (this.nativeProfileButton && this.nativeProfileButtonKey === key) return;
    this.destroyNativeProfileButton();
    this.nativeProfileButtonKey = key;
    this.nativeProfileButton = wx.createUserInfoButton({
      type: 'text',
      text: r.text,
      withCredentials: false,
      lang: 'zh_CN',
      style: {
        left: screen.left,
        top: screen.top,
        width: screen.width,
        height: screen.height,
        lineHeight: screen.height,
        backgroundColor: 'rgba(0,0,0,0)',
        color: '#ffffff',
        textAlign: 'center',
        fontSize: Math.max(14, 22 * this.scale),
        borderRadius: Math.max(6, 10 * this.scale)
      }
    });
    this.nativeProfileButton.onTap((res) => {
      if (res && res.userInfo) {
        this.applyUserProfile(res.userInfo);
        this.destroyNativeProfileButton();
      } else {
        this.tryGetUserInfoFallback();
      }
    });
  }

  destroyNativeProfileButton() {
    if (!this.nativeProfileButton) return;
    try {
      this.nativeProfileButton.destroy();
    } catch (e) {}
    this.nativeProfileButton = null;
    this.nativeProfileButtonKey = '';
  }

  setButtonsForFrame() {
    this.buttons = [];
    this.profileButtonRect = null;
    if (this.screen === 'lobby') {
      this.makeButton('server', `服务器 ${this.config.serverUrl}`, 80, 250, 760, 58, () => this.openKeyboard('serverUrl'), 'gray');
      this.makeButton('room', `房间 ${this.config.roomId || '留空自动创建'}`, 80, 326, 360, 58, () => this.openKeyboard('roomId'), 'gray');
      this.makeButton('name', `昵称 ${this.config.name || '玩家'}`, 462, 326, 378, 58, () => this.openKeyboard('name'), 'gray');
      this.makeButton('create', '创建房间', 890, 250, 210, 72, () => this.createRoom(), 'amber');
      this.makeButton('join', '加入房间', 1122, 250, 170, 72, () => this.joinRoom(false), 'green');
      this.makeProfileButton(this.profileReady ? '更新头像' : '授权头像', 890, 338, 402, 58, 'blue');
      return;
    }
    if (this.screen === 'room') {
      this.makeProfileButton(this.profileReady ? '头像' : '授权头像', 580, 612, 128, 58, 'blue');
      this.makeButton('share', '分享', 726, 612, 128, 58, () => this.shareRoom(), 'green');
      this.makeButton('addAI', '加 AI', 872, 612, 128, 58, () => this.addAI(), 'blue');
      this.makeButton('switchSeat', '换座', 1018, 612, 128, 58, () => this.switchSeat(), 'gray');
      const start = this.makeButton('start', '开始', 1164, 612, 128, 58, () => this.startGame(), 'amber');
      start.enabled = !!this.isHost && !!this.roomState && this.roomState.player_count >= 4;
      return;
    }
    if (this.screen === 'game') {
      const canPlay = this.canPlayNow();
      this.makeProfileButton('头像', 1138, 414, 142, 56, 'gray');
      this.makeButton('share', '分享', 1138, 482, 142, 56, () => this.shareRoom(), 'blue');
      const play = this.makeButton('play', '出牌', 1138, 560, 142, 62, () => this.playCards(), 'green');
      const pass = this.makeButton('pass', '不出', 1138, 636, 142, 62, () => this.passTurn(), 'gray');
      play.enabled = canPlay && this.selected.size > 0;
      pass.enabled = canPlay && !this.gameState.is_free_play;
      if (this.result) {
        const next = this.makeButton('next', '下一局', 589, 542, 156, 58, () => this.startGame(), 'amber');
        next.enabled = !!this.isHost;
      }
      if (this.shouldAskCha()) {
        this.makeButton('chaYes', '叉', 557, 440, 100, 54, () => this.respondCha(true), 'red');
        this.makeButton('chaNo', '不叉', 677, 440, 100, 54, () => this.respondCha(false), 'gray');
      }
      if (this.shouldAskDian()) {
        this.makeButton('dianYes', '点', 557, 440, 100, 54, () => this.respondDian(true), 'blue');
        this.makeButton('dianNo', '不点', 677, 440, 100, 54, () => this.respondDian(false), 'gray');
      }
    }
  }

  handleTouchStart(event) {
    const touch = event.changedTouches && event.changedTouches[0];
    if (!touch) return;
    const p = this.toLogical(touch);
    for (let i = this.buttons.length - 1; i >= 0; i--) {
      const btn = this.buttons[i];
      if (btn.contains(p.x, p.y)) {
        btn.onTap();
        return;
      }
    }
    if (this.screen === 'game' && !this.result && !this.shouldAskCha() && !this.shouldAskDian()) {
      const index = this.cardIndexAt(p.x, p.y);
      if (index >= 0) {
        this.touchSelectMode = !this.selected.has(index);
        this.setCardSelected(index, this.touchSelectMode);
        this.draggingCards = true;
      }
    }
  }

  handleTouchMove(event) {
    if (!this.draggingCards) return;
    const touch = event.changedTouches && event.changedTouches[0];
    if (!touch) return;
    const p = this.toLogical(touch);
    const index = this.cardIndexAt(p.x, p.y);
    if (index >= 0) this.setCardSelected(index, this.touchSelectMode);
  }

  cardIndexAt(x, y) {
    for (let i = this.cardRects.length - 1; i >= 0; i--) {
      const rect = this.cardRects[i];
      if (x >= rect.x && x <= rect.x + rect.w && y >= rect.y && y <= rect.y + rect.h) {
        return rect.index;
      }
    }
    return -1;
  }

  setCardSelected(index, selected) {
    if (selected) this.selected.add(index);
    else this.selected.delete(index);
  }

  openKeyboard(field) {
    this.editField = field;
    const titles = { serverUrl: '输入服务器地址', roomId: '输入房间号', name: '输入昵称' };
    const maxLength = field === 'serverUrl' ? 120 : 20;
    wx.showToast({ title: titles[field], icon: 'none', duration: 1200 });
    wx.showKeyboard({
      defaultValue: String(this.config[field] || ''),
      maxLength,
      multiple: false,
      confirmHold: false
    });
  }

  commitKeyboard(value) {
    if (!this.editField) return;
    const field = this.editField;
    this.editField = null;
    if (field === 'serverUrl') this.config[field] = normalizeServerUrl(value);
    else this.config[field] = String(value || '').trim();
    saveConfig(this.config);
    try {
      wx.hideKeyboard();
    } catch (e) {}
  }

  createRoom() {
    this.config.serverUrl = normalizeServerUrl(this.config.serverUrl);
    this.showToast('正在创建房间');
    this.requestJson('/api/rooms', 'POST',
      this.config.roomId ? { room_id: this.config.roomId } : {},
      (data) => {
        this.config.roomId = data.room_id || this.config.roomId || padRoomId();
        saveConfig(this.config);
        this.joinRoom(true);
      },
      (err) => {
        if (err && err.error === 'room_exists') {
          this.joinRoom(false);
          return;
        }
        this.showToast('无法访问服务器 REST 接口，请检查域名或关闭URL校验', 3200);
      });
  }

  joinRoom(isHost) {
    this.config.serverUrl = normalizeServerUrl(this.config.serverUrl);
    this.config.roomId = String(this.config.roomId || '').trim();
    this.config.name = String(this.config.name || '玩家').trim() || '玩家';
    if (!this.config.roomId) {
      this.createRoom();
      return;
    }
    this.isHost = !!isHost;
    saveConfig(this.config);
    this.socket.connect(this.config.serverUrl);
    this.socket.emit('join_room', {
      room_id: this.config.roomId,
      name: this.config.name,
      is_host: !!isHost,
      avatar_url: this.config.avatarUrl || '',
      session_token: this.config.sessionToken || ''
    });
    this.showToast('正在加入房间');
  }

  addAI() {
    this.socket.emit('add_ai', { model: 'rule' });
    this.showToast('正在添加 AI');
  }

  switchSeat() {
    const target = this.mySeat >= 0 ? (this.mySeat + 1) % 4 : 0;
    this.socket.emit('switch_seat', { seat: target });
  }

  startGame() {
    this.result = null;
    this.selected.clear();
    this.socket.emit('start_game', {});
  }

  playCards() {
    if (!this.canPlayNow() || !this.selected.size) return;
    const card_indices = Array.from(this.selected).sort((a, b) => a - b);
    this.pendingPlayIndices = card_indices;
    this.socket.emit('play_cards', { card_indices });
  }

  passTurn() {
    if (!this.canPlayNow() || this.gameState.is_free_play) return;
    this.socket.emit('pass_turn', {});
  }

  respondCha(value) {
    this.socket.emit('respond_cha', { do_cha: !!value });
  }

  respondDian(value) {
    this.socket.emit('respond_dian', { do_dian: !!value });
  }

  canPlayNow() {
    return this.gameState
      && this.gameState.phase === 'playing'
      && this.gameState.current_player_seat === this.mySeat;
  }

  shouldAskCha() {
    return this.gameState
      && this.gameState.phase === 'cha_asking'
      && this.gameState.cha_asking_seat === this.mySeat;
  }

  shouldAskDian() {
    return this.gameState
      && this.gameState.phase === 'dian_asking'
      && this.gameState.dian_asking_seat === this.mySeat;
  }

  getMe() {
    if (!this.gameState) return null;
    return (this.gameState.players || []).find((p) => p && p.seat === this.mySeat) || null;
  }

  syncHandSelection() {
    const me = this.getMe();
    const length = me && me.hand ? me.hand.length : 0;
    Array.from(this.selected).forEach((index) => {
      if (index < 0 || index >= length) this.selected.delete(index);
    });
  }

  addActionLog(data) {
    if (!data || !data.action) return;
    const player = this.findPlayer(data.seat);
    const name = player ? player.name : `座位${data.seat}`;
    let text = '';
    if (data.action === 'game_started') text = `开局，级牌 ${data.level_rank}`;
    else if (data.action === 'pass') text = `${name} 不出`;
    else if (data.action === 'cha') text = `${name} 叉牌 ${shortCards(data.cards)}`;
    else if (data.action === 'cha_pass') text = `${name} 不叉`;
    else if (data.action === 'dian') text = `${name} 点牌 ${shortCards(data.cards)}`;
    else if (data.action === 'dian_pass') text = `${name} 不点`;
    else if (data.cards && data.cards.length) text = `${name} 出 ${shortCards(data.cards)}`;
    if (data.player_finished) text += '，出完';
    if (!text) return;
    this.logs.unshift({ text, time: now() });
    this.logs = this.logs.slice(0, 7);
  }

  findPlayer(seat) {
    const source = this.gameState || this.roomState || {};
    return (source.players || []).find((p) => p && p.seat === seat) || null;
  }

  showToast(message, duration = 1800) {
    this.toast = { message, expires: now() + duration };
  }

  render() {
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.viewWidth, this.viewHeight);
    this.setButtonsForFrame();

    ctx.save();
    ctx.translate(this.offsetX, this.offsetY);
    ctx.scale(this.scale, this.scale);
    this.drawBackground(ctx);
    if (this.screen === 'lobby') this.drawLobby(ctx);
    else if (this.screen === 'room') this.drawRoom(ctx);
    else this.drawGame(ctx);
    this.buttons.forEach((btn) => btn.draw(ctx));
    this.drawToast(ctx);
    ctx.restore();
    this.syncNativeProfileButton();
  }

  drawBackground(g) {
    const gradient = g.createLinearGradient(0, 0, BASE_WIDTH, BASE_HEIGHT);
    gradient.addColorStop(0, '#123726');
    gradient.addColorStop(0.52, '#0d2b28');
    gradient.addColorStop(1, '#221f32');
    g.fillStyle = gradient;
    g.fillRect(0, 0, BASE_WIDTH, BASE_HEIGHT);

    g.save();
    g.translate(BASE_WIDTH / 2, 372);
    g.scale(1.15, 0.58);
    g.beginPath();
    g.arc(0, 0, 440, 0, Math.PI * 2);
    g.fillStyle = '#14523b';
    g.fill();
    g.lineWidth = 7;
    g.strokeStyle = '#d9b96a';
    g.stroke();
    g.lineWidth = 2;
    g.strokeStyle = 'rgba(255,255,255,0.18)';
    g.stroke();
    g.restore();
  }

  getAvatarImage(url) {
    if (!url) return null;
    if (this.imageCache[url]) return this.imageCache[url];
    const image = wx.createImage ? wx.createImage() : new Image();
    const record = { image, loaded: false, failed: false };
    image.onload = () => {
      record.loaded = true;
    };
    image.onerror = () => {
      record.failed = true;
    };
    image.src = url;
    this.imageCache[url] = record;
    return record;
  }

  drawAvatar(g, player, x, y, r) {
    const url = player && player.avatar_url ? player.avatar_url : '';
    const record = this.getAvatarImage(url);
    g.save();
    g.beginPath();
    g.arc(x, y, r, 0, Math.PI * 2);
    g.closePath();
    if (record && record.loaded && !record.failed) {
      g.clip();
      g.drawImage(record.image, x - r, y - r, r * 2, r * 2);
    } else {
      g.fillStyle = player ? '#2475d6' : '#44515b';
      g.fill();
      const label = player && player.name ? String(player.name).slice(0, 1) : '?';
      drawText(g, label, x, y + 1, Math.max(16, r), '#ffffff', 'center', 'bold');
    }
    g.restore();
    g.lineWidth = 2;
    g.strokeStyle = player && player.online === false ? '#ff9aa2' : '#ffd66b';
    g.beginPath();
    g.arc(x, y, r, 0, Math.PI * 2);
    g.stroke();
  }

  drawLobby(g) {
    drawText(g, '4A4', 88, 116, 96, '#ffd66b', 'left', 'bold');
    drawText(g, '微信横屏小游戏', 92, 190, 30, '#d7e8d9');
    drawText(g, '服务器、房间号和昵称可点按修改', 92, 220, 22, '#9db7a5');
    this.drawAvatar(g, {
      name: this.config.name,
      avatar_url: this.config.avatarUrl,
      online: this.loginReady
    }, 118, 548, 28);
    drawText(g, this.loginReady ? '微信登录已完成' : '正在微信登录...', 158, 536, 22, this.loginReady ? '#63e6a4' : '#ffd66b');
    drawText(g, this.profileReady ? '已使用微信头像' : '点击“授权头像”显示微信头像', 158, 568, 20, this.profileReady ? '#d7e8d9' : '#ffd66b');

    roundRect(g, 872, 356, 420, 132, 16);
    g.fillStyle = 'rgba(4,14,18,0.55)';
    g.fill();
    g.strokeStyle = 'rgba(255,255,255,0.16)';
    g.stroke();
    drawText(g, '连接说明', 900, 392, 28, '#ffd66b', 'left', 'bold');
    drawText(g, '开发者工具可关闭合法域名校验。', 900, 432, 21, '#d7e8d9');
    drawText(g, '真机需配置 request 和 socket 域名。', 900, 462, 21, '#d7e8d9');
    drawText(g, '示例：http://局域网IP:5000', 900, 492, 21, '#9db7a5');
  }

  drawRoom(g) {
    const state = this.roomState || {};
    drawText(g, `房间 ${state.room_id || this.config.roomId}`, 72, 64, 38, '#ffd66b', 'left', 'bold');
    drawText(g, `我的座位 ${this.mySeat >= 0 ? this.mySeat : '-'}    ${this.isHost ? '房主' : '玩家'}`, 72, 104, 22, '#d7e8d9');
    const levels = state.team_levels || {};
    drawText(g, `A队级别 ${levels[0] || levels['0'] || '3'}    B队级别 ${levels[1] || levels['1'] || '3'}    台上 ${state.on_stage_team === 1 ? 'B队' : 'A队'}`, 72, 138, 22, '#9db7a5');

    this.drawSeat(g, 190, 300, 0, state.players && state.players[0], this.mySeat === 0);
    this.drawSeat(g, 478, 300, 1, state.players && state.players[1], this.mySeat === 1);
    this.drawSeat(g, 766, 300, 2, state.players && state.players[2], this.mySeat === 2);
    this.drawSeat(g, 1054, 300, 3, state.players && state.players[3], this.mySeat === 3);

    drawText(g, `人数 ${state.player_count || 0}/4`, 72, 628, 24, '#d7e8d9');
    if (!this.isHost) {
      drawText(g, '等待房主开始游戏', 72, 664, 24, '#ffd66b');
    } else if ((state.player_count || 0) < 4) {
      drawText(g, '需要 4 名玩家，可点“加 AI”补齐', 72, 664, 24, '#ffd66b');
    } else {
      drawText(g, '人员已满，可以开始', 72, 664, 24, '#63e6a4');
    }
  }

  drawSeat(g, x, y, seat, player, me) {
    roundRect(g, x - 120, y - 76, 240, 152, 16);
    g.fillStyle = me ? 'rgba(255,214,107,0.22)' : 'rgba(4,14,18,0.58)';
    g.fill();
    g.lineWidth = 2;
    g.strokeStyle = me ? '#ffd66b' : 'rgba(255,255,255,0.16)';
    g.stroke();
    drawText(g, `座位 ${seat}  ${seat % 2 === 0 ? 'A队' : 'B队'}`, x, y - 40, 24, '#ffd66b', 'center', 'bold');
    this.drawAvatar(g, player, x - 72, y + 8, 24);
    drawText(g, player ? player.name : '空位', x + 18, y + 4, 24, player ? '#ffffff' : '#9db7a5', 'center', 'bold');
    const status = player && player.online === false ? '离线' : (player && player.is_ai ? 'AI' : (me ? '我' : ''));
    drawText(g, status, x, y + 42, 21, player && player.online === false ? '#ff9aa2' : '#9db7a5', 'center');
  }

  drawGame(g) {
    if (!this.gameState) {
      drawText(g, '等待牌局状态...', BASE_WIDTH / 2, 370, 30, '#d7e8d9', 'center');
      return;
    }
    this.drawGameHeader(g);
    this.drawPlayers(g);
    this.drawPlayHistory(g);
    this.drawHand(g);
    this.drawTurnHint(g);
    if (this.shouldAskCha()) this.drawChoiceDialog(g, '是否叉牌？', '叉牌会打出两张指定级牌');
    if (this.shouldAskDian()) this.drawChoiceDialog(g, '是否点牌？', '点牌会打出一张指定级牌');
    if (this.result) this.drawResult(g);
  }

  drawGameHeader(g) {
    const st = this.gameState;
    const phaseName = {
      playing: '出牌中',
      cha_asking: '叉牌',
      dian_asking: '点牌',
      round_end: '结束',
      waiting: '等待'
    }[st.phase] || st.phase;
    drawText(g, `4A4  房间 ${this.config.roomId}`, 38, 34, 24, '#d7e8d9');
    drawText(g, `级牌 ${st.level_rank}    台上 ${st.on_stage_team === 1 ? 'B队' : 'A队'}    ${phaseName}`, 38, 70, 24, '#ffd66b', 'left', 'bold');
  }

  drawPlayers(g) {
    const seats = [this.mySeat, (this.mySeat + 1) % 4, (this.mySeat + 2) % 4, (this.mySeat + 3) % 4];
    const spots = [
      { x: 667, y: 516, align: 'center' },
      { x: 1120, y: 268, align: 'center' },
      { x: 667, y: 146, align: 'center' },
      { x: 214, y: 268, align: 'center' }
    ];
    seats.forEach((seat, idx) => {
      const p = this.findPlayer(seat);
      const spot = spots[idx];
      this.drawPlayerBadge(g, spot.x, spot.y, seat, p, seat === this.gameState.current_player_seat);
    });
  }

  drawPlayerBadge(g, x, y, seat, player, active) {
    roundRect(g, x - 94, y - 34, 188, 68, 12);
    g.fillStyle = active ? 'rgba(255,214,107,0.26)' : 'rgba(3,12,16,0.72)';
    g.fill();
    g.strokeStyle = active ? '#ffd66b' : 'rgba(255,255,255,0.14)';
    g.stroke();
    const name = player ? player.name : `座位${seat}`;
    const count = player ? player.hand_size : 0;
    this.drawAvatar(g, player, x - 66, y, 22);
    drawText(g, `${name}${seat === this.mySeat ? ' / 我' : ''}`, x + 14, y - 10, 18, '#ffffff', 'center', 'bold');
    const status = player && player.online === false ? '  离线' : '';
    drawText(g, `${seat % 2 === 0 ? 'A队' : 'B队'}  ${count}张${player && player.finished ? '  已出完' : ''}${status}`, x + 14, y + 18, 16, player && player.online === false ? '#ff9aa2' : '#a8c7b4', 'center');
  }

  drawPlayHistory(g) {
    const history = this.gameState.play_history || [];
    const latest = {};
    history.slice().reverse().forEach((item) => {
      if (latest[item.seat] === undefined && ['play', 'pass', 'cha', 'dian'].indexOf(item.action) >= 0) {
        latest[item.seat] = item;
      }
    });
    Object.keys(latest).forEach((seatText) => {
      const seat = Number(seatText);
      if (seat === this.mySeat) return;
      const item = latest[seat];
      const pos = this.playSpotForSeat(seat);
      if (!pos) return;
      const label = item.action === 'pass' ? '不出' : shortCards(item.cards, 8);
      drawText(g, label, pos.x, pos.y, 20, item.action === 'pass' ? '#9db7a5' : '#ffffff', 'center');
    });

    roundRect(g, 38, 548, 330, 152, 12);
    g.fillStyle = 'rgba(3,12,16,0.52)';
    g.fill();
    drawText(g, '牌局记录', 58, 576, 21, '#ffd66b', 'left', 'bold');
    this.logs.forEach((item, index) => {
      drawText(g, item.text, 58, 606 + index * 20, 17, '#d7e8d9');
    });
  }

  playSpotForSeat(seat) {
    const rel = (seat - this.mySeat + 4) % 4;
    return [
      { x: 667, y: 430 },
      { x: 996, y: 360 },
      { x: 667, y: 224 },
      { x: 338, y: 360 }
    ][rel];
  }

  drawHand(g) {
    this.cardRects = [];
    const me = this.getMe();
    const hand = me && me.hand ? me.hand : [];
    if (!hand.length) return;
    const n = hand.length;
    const cardW = n > 24 ? 58 : n > 20 ? 62 : 68;
    const cardH = Math.floor(cardW * 1.42);
    const maxSpan = 910;
    const gap = n <= 1 ? 0 : Math.min(cardW * 0.66, (maxSpan - cardW) / (n - 1));
    const total = cardW + gap * (n - 1);
    const startX = 667 - total / 2;
    const baseY = 624;
    hand.forEach((card, index) => {
      const x = startX + index * gap;
      const y = baseY - (this.selected.has(index) ? 28 : 0);
      this.drawCard(g, card, x, y, cardW, cardH, index);
      this.cardRects.push({ x, y, w: cardW, h: cardH, index });
    });
  }

  drawCard(g, card, x, y, w, h, index) {
    roundRect(g, x, y, w, h, 9);
    g.fillStyle = '#fffaf0';
    g.fill();
    g.lineWidth = this.selected.has(index) ? 4 : 1.6;
    g.strokeStyle = this.selected.has(index) ? '#ffd66b' : '#c9b98c';
    g.stroke();
    const red = isRedCard(card);
    const rank = cardRank(card);
    const suit = suitSymbol(cardSuit(card));
    if (rank === 'RJ' || rank === 'BJ') {
      drawText(g, rank === 'RJ' ? '大' : '小', x + w / 2, y + h * 0.38, Math.floor(w * 0.44), red ? '#d7263d' : '#222', 'center', 'bold');
      drawText(g, '王', x + w / 2, y + h * 0.64, Math.floor(w * 0.44), red ? '#d7263d' : '#222', 'center', 'bold');
    } else {
      drawText(g, rank, x + w / 2, y + h * 0.34, Math.floor(w * 0.38), red ? '#d7263d' : '#1c2530', 'center', 'bold');
      drawText(g, suit, x + w / 2, y + h * 0.65, Math.floor(w * 0.4), red ? '#d7263d' : '#1c2530', 'center', 'bold');
    }
  }

  drawTurnHint(g) {
    const st = this.gameState;
    let text = '等待其他玩家';
    if (this.canPlayNow()) text = st.is_free_play ? '轮到你领出' : '轮到你出牌';
    else {
      const current = this.findPlayer(st.current_player_seat);
      if (current) text = `等待 ${current.name}`;
    }
    drawText(g, text, 667, 552, 24, this.canPlayNow() ? '#ffd66b' : '#a8c7b4', 'center', 'bold');
  }

  drawChoiceDialog(g, title, body) {
    g.save();
    g.fillStyle = 'rgba(0,0,0,0.52)';
    g.fillRect(0, 0, BASE_WIDTH, BASE_HEIGHT);
    roundRect(g, 462, 270, 410, 244, 18);
    g.fillStyle = '#10211f';
    g.fill();
    g.strokeStyle = '#ffd66b';
    g.stroke();
    drawText(g, title, 667, 326, 32, '#ffd66b', 'center', 'bold');
    drawText(g, body, 667, 374, 21, '#d7e8d9', 'center');
    g.restore();
  }

  drawResult(g) {
    const data = this.result || {};
    g.save();
    g.fillStyle = 'rgba(0,0,0,0.62)';
    g.fillRect(0, 0, BASE_WIDTH, BASE_HEIGHT);
    roundRect(g, 402, 150, 530, 450, 18);
    g.fillStyle = '#111f24';
    g.fill();
    g.strokeStyle = '#ffd66b';
    g.stroke();
    const winner = data.winner_team === 0 ? 'A队获胜' : data.winner_team === 1 ? 'B队获胜' : '平局';
    drawText(g, winner, 667, 210, 40, '#ffd66b', 'center', 'bold');
    const lines = [];
    lines.push(`A队：${this.resultName(data.team0_result)}`);
    lines.push(`B队：${this.resultName(data.team1_result)}`);
    if (data.upgrade > 0) lines.push(`升级 ${data.upgrade} 级`);
    if (data.zhi_j) lines.push('直J，退回打3');
    if (data.zhi_a) lines.push('直A，退回打J');
    const levels = data.new_level || {};
    lines.push(`新级别 A队 ${levels[0] || levels['0'] || '-'} / B队 ${levels[1] || levels['1'] || '-'}`);
    lines.forEach((line, index) => drawText(g, line, 667, 282 + index * 36, 24, '#d7e8d9', 'center'));
    if (!this.isHost) drawText(g, '等待房主开启下一局', 667, 565, 20, '#9db7a5', 'center');
    g.restore();
  }

  resultName(value) {
    return { quan_dong: '全洞', ban_dong: '半洞', lose: '未赢' }[value] || value || '-';
  }

  drawToast(g) {
    if (!this.toast || this.toast.expires < now()) return;
    const width = clamp(String(this.toast.message).length * 20 + 70, 220, 820);
    roundRect(g, (BASE_WIDTH - width) / 2, 96, width, 48, 24);
    g.fillStyle = 'rgba(0,0,0,0.72)';
    g.fill();
    drawText(g, this.toast.message, BASE_WIDTH / 2, 120, 22, '#ffffff', 'center');
  }
}
