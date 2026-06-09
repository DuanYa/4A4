class WaitingScene extends Phaser.Scene {
    constructor() { super({ key: 'WaitingScene' }); }

    init(data) {
        this.roomId = data.roomId;
        this.mySeat = data.seat;
        this.isHost = false;
        this.aiMode = 'rule';
        this.checkpoints = ['default_rl.pt'];
        this.checkpointIndex = 0;
        this._initialRoomState = data.roomState || null;
    }

    create() {
        makeLayout(this);
        MobileUI.bg(this);
        MobileUI.text(this, 667, 56, '牌桌 ' + this.roomId, 36, '#ffd66b', 'bold');
        this.seatTitle = MobileUI.text(this, 667, 96, '座位 ' + (this.mySeat + 1) + ' · ' + (this.mySeat % 2 === 0 ? 'A队' : 'B队'), 21, '#d8e8dc');
        this.createShareButton();
        this.createSeatLine();
        this.createInfoBar();
        this.createActionDock();
        this.setupWebSocketHandlers();
        this.loadCheckpoints();
        if (this._initialRoomState) {
            this.updateRoomState(this._initialRoomState);
            this._initialRoomState = null;
        }
    }

    createShareButton() {
        this.shareButton = MobileUI.button(this, 1220, 56, 170, 54, '复制邀请链接', 0x305c83, () => {});

        const domBtn = document.createElement('button');
        domBtn.textContent = '复制邀请链接';
        domBtn.style.cssText = 'position:fixed;z-index:40;border:none;background:#305c83;color:#fff;font-family:Microsoft YaHei,sans-serif;font-weight:bold;border-radius:27px;cursor:pointer;border:2px solid rgba(255,255,255,0.22);outline:none;-webkit-tap-highlight-color:transparent;pointer-events:auto;';
        domBtn.addEventListener('pointerdown', function (e) {
            e.preventDefault();
            e.stopPropagation();
        });
        domBtn.addEventListener('click', function (e) {
            e.preventDefault();
            e.stopPropagation();
            ShareHelper.share(this._roomId);
            MobileUI.toast(this._scene, '分享链接已复制到剪贴板', 0x1f6f46);
        }.bind({ _roomId: this.roomId, _scene: this }));
        document.body.appendChild(domBtn);
        this._shareDomBtn = domBtn;

        this.add.text(this.layout.x(1105), this.layout.y(80),
            '🔗', { fontSize: '18px', fontFamily: 'Arial', color: '#ffffff' }).setOrigin(0.5);

        this._positionShareDomBtn();
        this.scale.on('resize', this._positionShareDomBtn, this);
    }

    _positionShareDomBtn() {
        if (!this._shareDomBtn || !this.layout) return;
        const rect = this.layout.domRect(1220 - 85, 56 - 27, 170, 54);
        const fs = Math.max(12, Math.round(rect.h * 0.38));
        this._shareDomBtn.style.left = Math.round(rect.x) + 'px';
        this._shareDomBtn.style.top = Math.round(rect.y) + 'px';
        this._shareDomBtn.style.width = Math.round(rect.w) + 'px';
        this._shareDomBtn.style.height = Math.round(rect.h) + 'px';
        this._shareDomBtn.style.fontSize = fs + 'px';
    }

    createSeatLine() {
        this.seatCards = [];
        const xs = [215, 515, 815, 1115];
        for (let seat = 0; seat < 4; seat++) {
            const c = this.add.container(this.layout.x(xs[seat]), this.layout.y(275));
            const bg = this.add.graphics();
            bg.fillStyle(0x102d22, 0.82);
            bg.fillRoundedRect(-this.layout.s(118), -this.layout.s(86), this.layout.s(236), this.layout.s(172), this.layout.s(26));
            bg.lineStyle(this.layout.s(3), seat === this.mySeat ? 0x66ff99 : 0xffd66b, seat === this.mySeat ? 0.9 : 0.26);
            bg.strokeRoundedRect(-this.layout.s(118), -this.layout.s(86), this.layout.s(236), this.layout.s(172), this.layout.s(26));
            const avatar = this.add.graphics();
            avatar.fillStyle(seat % 2 === 0 ? 0x2b8a57 : 0x8a5b2b, 0.98);
            avatar.fillCircle(0, -this.layout.s(30), this.layout.s(46));
            const face = this.add.text(0, -this.layout.s(33), seat % 2 === 0 ? 'A' : 'B', {
                fontSize: Math.floor(this.layout.s(34)) + 'px', fontFamily: 'Arial Black', color: '#ffffff'
            }).setOrigin(0.5);
            const name = this.add.text(0, this.layout.s(35), '等待中', {
                fontSize: Math.floor(this.layout.s(23)) + 'px', fontFamily: 'Microsoft YaHei', color: '#ffffff', fontStyle: 'bold'
            }).setOrigin(0.5);
            const sub = this.add.text(0, this.layout.s(68), '座位' + (seat + 1), {
                fontSize: Math.floor(this.layout.s(17)) + 'px', fontFamily: 'Microsoft YaHei', color: '#b9d2c2'
            }).setOrigin(0.5);
            c.add([bg, avatar, face, name, sub]);
            const hit = this.add.rectangle(0, 0, this.layout.s(236), this.layout.s(172), 0xffffff, 0).setOrigin(0.5);
            hit.setInteractive({ useHandCursor: true });
            hit.on('pointerdown', () => this.switchSeat(seat));
            c.add(hit);
            this.seatCards[seat] = { c, name, sub, bg };
        }
    }

    createInfoBar() {
        MobileUI.panel(this, 667, 442, 900, 88, 0.72);
        this.levelText = MobileUI.text(this, 435, 442, 'A队 3', 24, '#ffd66b', 'bold');
        this.levelTextB = MobileUI.text(this, 667, 442, 'B队 3', 24, '#ffd66b', 'bold');
        this.stageText = MobileUI.text(this, 900, 442, '台上 A队', 24, '#ffffff');
        this.countText = MobileUI.text(this, 667, 510, '等待玩家加入 1/4', 21, '#b9d2c2');
    }

    createActionDock() {
        MobileUI.panel(this, 667, 628, 1080, 150, 0.84);
        this.modelButton = MobileUI.button(this, 250, 600, 220, 58, '规则AI', 0x2d7f52, () => this.toggleModel());
        this.ckptButton = MobileUI.button(this, 520, 600, 270, 58, 'default_rl.pt', 0x305c83, () => this.nextCheckpoint());
        this.addAIButton = MobileUI.button(this, 815, 600, 220, 66, '添加AI', 0x27ae60, () => this.addAI());
        this.startButton = MobileUI.button(this, 1085, 600, 220, 66, '开始', 0xe6a800, () => this.startGame());
        this.startButton.setEnabled(false);
    }

    toggleModel() {
        this.aiMode = this.aiMode === 'rule' ? 'rl' : 'rule';
        this.modelButton.text.setText(this.aiMode === 'rule' ? '规则AI' : '强化AI');
    }

    nextCheckpoint() {
        if (!this.checkpoints.length) return;
        this.checkpointIndex = (this.checkpointIndex + 1) % this.checkpoints.length;
        this.ckptButton.text.setText(this.checkpoints[this.checkpointIndex]);
    }

    loadCheckpoints() {
        fetch('/api/checkpoints')
            .then(r => r.json())
            .then(data => {
                this.checkpoints = data.checkpoints && data.checkpoints.length ? data.checkpoints : ['default_rl.pt'];
                this.checkpointIndex = 0;
                this.ckptButton.text.setText(this.checkpoints[0]);
            })
            .catch(err => AppLogger.warn('WaitingScene', 'checkpoint加载失败', err));
    }

    addAI() {
        if (!this.isHost) return MobileUI.toast(this, '只有房主可以添加AI', 0x8f2f2f);
        const model = this.aiMode === 'rule' ? 'rule' : this.checkpoints[this.checkpointIndex];
        this.addAIButton.setEnabled(false);
        wsManager.send('add_ai', { model });
    }

    startGame() {
        if (!this.isHost) return MobileUI.toast(this, '只有房主可以开始', 0x8f2f2f);
        wsManager.send('start_game', {});
    }

    switchSeat(seat) {
        if (seat === this.mySeat) return;
        wsManager.send('switch_seat', { seat });
    }

    updateRoomState(data) {
        const myPlayer = data.players.find(p => p && p.player_id === wsManager.socket.id);
        this.isHost = data.host_player_id === wsManager.socket.id;
        if (myPlayer && myPlayer.seat !== this.mySeat) this.mySeat = myPlayer.seat;
        if (this.seatTitle) this.seatTitle.setText('座位 ' + (this.mySeat + 1) + ' · ' + (this.mySeat % 2 === 0 ? 'A队' : 'B队'));
        data.players.forEach((p, seat) => {
            const item = this.seatCards[seat];
            if (!item) return;
            if (p) {
                item.name.setText(p.name + (p.player_id === data.host_player_id ? ' 房主' : ''));
                item.name.setColor(p.seat === this.mySeat ? '#66ff99' : '#ffffff');
                item.sub.setText((seat % 2 === 0 ? 'A队' : 'B队') + ' · 已就座');
            } else {
                item.name.setText('等待中');
                item.name.setColor('#ffffff');
                item.sub.setText((seat % 2 === 0 ? 'A队' : 'B队') + ' · 空位');
            }
        });
        this.levelText.setText('A队 ' + (data.team_levels['0'] || '3'));
        this.levelTextB.setText('B队 ' + (data.team_levels['1'] || '3'));
        this.stageText.setText('台上 ' + (data.on_stage_team === 0 ? 'A队' : 'B队'));
        this.countText.setText('等待玩家加入 ' + data.player_count + '/4' + (this.isHost ? ' · 你是房主' : ' · 等待房主'));
        this.addAIButton.text.setText('添加AI');
        this.startButton.setEnabled(this.isHost && data.player_count >= 4 && !data.game_active);
        this.addAIButton.setEnabled(this.isHost && data.player_count < 4 && !data.game_active);
    }

    setupWebSocketHandlers() {
        this.roomStateHandler = data => this.updateRoomState(data);
        this.gameActionHandler = data => {
            if (data.action === 'game_started') {
                if (this._shareDomBtn) {
                    this._shareDomBtn.style.display = 'none';
                }
                this._gameStateWaiter = state => {
                    wsManager.off('game_state', this._gameStateWaiter);
                    this._gameStateWaiter = null;
                    this.scene.start('GameScene', {
                        roomId: this.roomId,
                        seat: this.mySeat,
                        levelRank: data.level_rank,
                        isHost: this.isHost,
                        initialGameState: state
                    });
                };
                wsManager.on('game_state', this._gameStateWaiter);
            }
        };
        this.seatChangedHandler = data => {
            this.mySeat = data.seat;
            if (this.seatTitle) this.seatTitle.setText('座位 ' + (this.mySeat + 1) + ' · ' + (this.mySeat % 2 === 0 ? 'A队' : 'B队'));
            MobileUI.toast(this, '已切换到座位 ' + (this.mySeat + 1), 0x1f6f46);
        };
        this.errorHandler = data => MobileUI.toast(this, data.message || '操作失败', 0x8f2f2f);
        wsManager.on('room_state', this.roomStateHandler);
        wsManager.on('game_action', this.gameActionHandler);
        wsManager.on('seat_changed', this.seatChangedHandler);
        wsManager.on('error', this.errorHandler);
    }

    shutdown() {
        if (this._gameStateWaiter) {
            wsManager.off('game_state', this._gameStateWaiter);
            this._gameStateWaiter = null;
        }
        if (this._shareDomBtn) {
            this._shareDomBtn.remove();
            this._shareDomBtn = null;
        }
        this.scale.off('resize', this._positionShareDomBtn, this);
        wsManager.off('room_state', this.roomStateHandler);
        wsManager.off('game_action', this.gameActionHandler);
        wsManager.off('seat_changed', this.seatChangedHandler);
        wsManager.off('error', this.errorHandler);
    }
}
