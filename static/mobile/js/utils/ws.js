class WSManager {
    constructor() {
        this.socket = null;
        this.handlers = {};
        this.connected = false;
    }

    connect() {
        AppLogger.info('WS', '开始连接WebSocket');
        this.socket = io({
            transports: ['websocket', 'polling']
        });

        this.socket.on('connect', () => {
            this.connected = true;
            AppLogger.info('WS', 'WebSocket已连接', { id: this.socket.id });
            this._emit('connected');
        });

        this.socket.on('disconnect', () => {
            this.connected = false;
            AppLogger.warn('WS', 'WebSocket已断开');
            this._emit('disconnected');
        });

        const events = [
            'joined', 'error', 'room_state', 'player_left',
            'game_state', 'game_action', 'round_end'
        ];
        events.forEach(evt => {
            this.socket.on(evt, data => {
                AppLogger.info('WS', '收到事件 ' + evt, data);
                this._emit(evt, data);
            });
        });
    }

    on(event, handler) {
        if (!this.handlers[event]) this.handlers[event] = [];
        this.handlers[event].push(handler);
    }

    off(event, handler) {
        if (!this.handlers[event]) return;
        if (handler) {
            this.handlers[event] = this.handlers[event].filter(h => h !== handler);
        } else {
            this.handlers[event] = [];
        }
    }

    _emit(event, data) {
        (this.handlers[event] || []).forEach(h => h(data));
    }

    send(event, data, ack) {
        AppLogger.info('WS', '发送事件 ' + event, data || {});
        if (!this.socket) return;
        if (ack) this.socket.emit(event, data || {}, ack);
        else this.socket.emit(event, data || {});
    }
}

const wsManager = new WSManager();
