/**
 * WebSocket 客户端封装
 */
class WS {
    constructor() {
        this.socket = null;
        this.handlers = {};
    }

    connect() {
        AppLogger.info('WS', '开始连接WebSocket');
        this.socket = io({
            transports: ['websocket', 'polling']
        });

        this.socket.on('connect', () => {
            AppLogger.info('WS', 'WebSocket已连接', { id: this.socket.id });
            this._emit('connected');
        });

        this.socket.on('disconnect', () => {
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

    _emit(event, data) {
        (this.handlers[event] || []).forEach(h => h(data));
    }

    send(event, data) {
        AppLogger.info('WS', '发送事件 ' + event, data || {});
        if (this.socket) this.socket.emit(event, data || {});
    }
}

const ws = new WS();
