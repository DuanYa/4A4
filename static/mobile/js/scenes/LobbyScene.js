class LobbyScene extends Phaser.Scene {
    constructor() { super({ key: 'LobbyScene' }); }

    create() {
        makeLayout(this);
        AppLogger.info('LobbyScene', '进入横屏移动端大厅');
        MobileUI.bg(this);

        this.add.text(this.layout.x(260), this.layout.y(170), '4A4', {
            fontSize: Math.floor(this.layout.s(118)) + 'px',
            fontFamily: 'Arial Black, Microsoft YaHei',
            color: '#ffd66b',
            stroke: '#5b3000',
            strokeThickness: Math.floor(this.layout.s(8)),
            shadow: { offsetX: 0, offsetY: this.layout.s(8), color: '#000000', blur: this.layout.s(18), fill: true }
        }).setOrigin(0.5);
        MobileUI.text(this, 260, 268, '四幺四横屏牌桌', 34, '#f8f2cf', 'bold');
        MobileUI.text(this, 260, 328, '横屏 · 无遮挡 · 大触控 · 滑动选牌', 22, '#b9d2c2');

        MobileUI.panel(this, 915, 350, 640, 430, 0.82);
        MobileUI.text(this, 915, 178, '快速开局', 36, '#ffd66b', 'bold');
        MobileUI.text(this, 915, 230, '输入昵称和房间号，留空自动创建房间', 22, '#d8e8dc');

        this.nameInput = MobileUI.inputBox(this, 915, 310, 430, 58, '输入昵称', 8);
        this.roomInput = MobileUI.inputBox(this, 915, 400, 430, 58, '房间号，留空自动创建', 8);
        this.createdRoomId = null;
        this.quickButton = MobileUI.button(this, 915, 505, 420, 72, '创建 / 加入牌桌', 0xe6a800, () => this.joinOrCreateRoom());
        this.statusText = MobileUI.text(this, 915, 604, '', 24, '#ff7878');
        MobileUI.text(this, 667, 708, '移动横屏版与桌面版完全隔离 · Phaser 3 Canvas渲染', 19, '#7ea08a');
        this.setupWebSocketHandlers();
        this._tryAutoJoin();
    }

    _tryAutoJoin() {
        const shareRoomId = window.__shareRoomId;
        if (!shareRoomId) return;

        AppLogger.info('LobbyScene', '分享链接自动加入房间', { roomId: shareRoomId });
        this.roomInput.setValue(shareRoomId);

        if (!this.nameInput.value) {
            const randomSuffix = Math.floor(Math.random() * 9000 + 1000);
            this.nameInput.setValue('玩家' + randomSuffix);
        }

        if (wsManager.connected) {
            this.joinOrCreateRoom();
        } else {
            this.showStatus('正在连接服务器...', '#ffd66b');
            const onConnected = () => {
                wsManager.off('connected', onConnected);
                this.showStatus('正在进入牌桌...', '#ffd66b');
                this.joinOrCreateRoom();
            };
            wsManager.on('connected', onConnected);
        }
    }

    joinOrCreateRoom() {
        const name = (this.nameInput.value || '').trim() || '玩家';
        const roomId = (this.roomInput.value || '').trim();
        this.quickButton.setEnabled(false);
        this.showStatus('正在进入牌桌...', '#ffd66b');
        if (roomId) {
            this.createdRoomId = null;
            this._doJoinRoom(roomId, name, false);
            return;
        }
        fetch('/api/rooms', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    this.showStatus(data.error, '#ff7878');
                    this.quickButton.setEnabled(true);
                    return;
                }
                this.createdRoomId = data.room_id;
                this._doJoinRoom(data.room_id, name, true);
            })
            .catch(err => {
                AppLogger.error('LobbyScene', '创建房间失败', err);
                this.showStatus('网络连接失败', '#ff7878');
                this.quickButton.setEnabled(true);
            });
    }

    _doJoinRoom(roomId, name, isHost) {
        wsManager.send('join_room', { room_id: roomId, name, is_host: isHost });
    }

    showStatus(message, color) {
        this.statusText.setText(message);
        this.statusText.setColor(color || '#ffffff');
    }

    setupWebSocketHandlers() {
        this.joinedHandler = data => this.scene.start('WaitingScene', {
            roomId: data.room_id,
            seat: data.seat,
            roomState: data.room_state
        });
        this.errorHandler = data => {
            this.quickButton.setEnabled(true);
            this.showStatus(data.message || '加入失败', '#ff7878');
            MobileUI.toast(this, data.message || '加入失败', 0x8f2f2f);
        };
        wsManager.on('joined', this.joinedHandler);
        wsManager.on('error', this.errorHandler);
    }

    shutdown() {
        MobileUI.destroyNativeInputs(this);
        wsManager.off('joined', this.joinedHandler);
        wsManager.off('error', this.errorHandler);
    }
}
