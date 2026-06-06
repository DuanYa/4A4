"""
WebSocket 事件处理 - 游戏实时通信
支持AI玩家手动补位
"""
import threading
import time
import logging
from models.game import GamePhase
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask import request

socketio = None
room_manager = None
logger = logging.getLogger('ws')

# sid -> {'room_id': str, 'seat': int}
player_sessions = {}
session_lock = threading.Lock()

# room_id -> [AIPlayer实例列表]
ai_players = {}
ai_lock = threading.Lock()

ERR_MSG = {
    'not_playing': '当前不是出牌阶段',
    'not_your_turn': '还没轮到你出牌',
    'already_finished': '你已经出完牌了',
    'no_cards_selected': '请选择要出的牌',
    'invalid_index': '无效的牌索引',
    'invalid_hand': '不是合法的牌型',
    'no_double_straight_lead': '首家不可出双龙',
    'cannot_beat': '出的牌管不住上家',
    'not_cha_phase': '当前不是叉牌询问阶段',
    'not_dian_phase': '当前不是点牌询问阶段',
    'not_asked': '没有询问你',
    'not_enough_cards': '手牌不足',
    'no_card_to_dian': '手牌中没有牌来点',
    'must_play_as_leader': '你是首家，必须出牌',
}


def get_msg(code):
    return ERR_MSG.get(code, code)


def init_ws(sio, rm):
    global socketio, room_manager
    socketio = sio
    room_manager = rm
    register_events(sio)


def broadcast_state(room_id):
    room = room_manager.get(room_id)
    if not room or not room.current_game:
        logger.info('broadcast_state skipped: room_id=%s has_game=%s',
                    room_id, bool(room and room.current_game))
        return
    logger.info('broadcast_state: room_id=%s phase=%s current=%s history=%d',
                room_id, room.current_game.phase.value,
                room.current_game.current_player_seat,
                len(getattr(room.current_game, 'play_history', [])))
    for p in room.players:
        if p is None:
            continue
        sid = p.player_id
        state = room.current_game.get_state(
            for_seat=p.seat)
        socketio.emit('game_state', state, to=sid)


def broadcast_action(room_id, action_data):
    logger.info('broadcast_action: room_id=%s action=%s data=%s',
                room_id, action_data.get('action'), action_data)
    socketio.emit('game_action', action_data, to=room_id)


def _next_ai_name(room):
    """生成房间内不重复的AI名字"""
    existing = {p.name for p in room.players if p is not None}
    idx = 1
    while True:
        name = 'AI-%d' % idx
        if name not in existing:
            return name
        idx += 1


def _add_ai_player(room_id, room, model_name='rule'):
    """添加一个AI玩家到房间"""
    from models.ai_player import AIPlayer

    if room.is_full():
        return True

    bot = AIPlayer(_next_ai_name(room), model_name=model_name)
    try:
        bot.connect_and_join(room_id)
    except Exception:
        bot.disconnect()
        return False

    with ai_lock:
        if room_id not in ai_players:
            ai_players[room_id] = []
        ai_players[room_id].append(bot)

    for _ in range(100):
        if bot.seat >= 0:
            return True
        time.sleep(0.05)
    return bot.seat >= 0


def _fill_ai_players(room_id, room, model_name='rule'):
    """用AI玩家补满房间空位"""
    while not room.is_full():
        if not _add_ai_player(room_id, room, model_name):
            return False
    return True


def _cleanup_ai_players(room_id):
    """断开房间内所有AI玩家"""
    with ai_lock:
        bots = ai_players.pop(room_id, [])
    for bot in bots:
        bot.disconnect()


def register_events(sio):

    @sio.on('connect')
    def on_connect():
        logger.info('WS connect: sid=%s', request.sid)

    @sio.on('disconnect')
    def on_disconnect():
        sid = request.sid
        logger.info('WS disconnect: sid=%s', sid)
        with session_lock:
            info = player_sessions.pop(sid, None)
        if info:
            rid = info['room_id']
            logger.info('WS player session removed: sid=%s room_id=%s seat=%s',
                        sid, rid, info['seat'])
            room = room_manager.get(rid)
            if room:
                room.remove_player(sid)
                leave_room(rid, sid=sid)
                sio.emit('player_left', {
                    'seat': info['seat'],
                }, to=rid)
                sio.emit('room_state',
                         room.get_room_state(), to=rid)

    @sio.on('join_room')
    def on_join_room(data):
        sid = request.sid
        rid = data.get('room_id', '')
        name = data.get('name', 'Player')
        logger.info('WS join_room request: sid=%s room_id=%s name=%s', sid, rid, name)
        if rid not in room_manager:
            emit('error', {'message': '房间不存在'})
            return
        room = room_manager[rid]
        if room.is_full():
            emit('error', {'message': '房间已满'})
            return
        is_ai = bool(data.get('is_ai', False))
        seat = room.add_player(sid, name, is_ai=is_ai)
        if data.get('is_host') and not is_ai:
            room.host_player_id = sid
        if seat < 0:
            emit('error', {'message': '加入失败'})
            return
        with session_lock:
            player_sessions[sid] = {
                'room_id': rid, 'seat': seat}
        join_room(rid)
        logger.info('WS join_room success: sid=%s room_id=%s seat=%s name=%s',
                    sid, rid, seat, name)
        room_state = room.get_room_state()
        emit('joined', {'seat': seat, 'room_id': rid, 'room_state': room_state})
        sio.emit('room_state', room_state, to=rid)

    @sio.on('start_game')
    def on_start_game(data):
        sid = request.sid
        with session_lock:
            info = player_sessions.get(sid)
        if not info:
            emit('error', {'message': '未加入房间'})
            return
        rid = info['room_id']
        room = room_manager.get(rid)
        if not room:
            emit('error', {'message': '房间不存在'})
            return
        if not room.is_full():
            logger.info('WS start_game failed not full: room_id=%s count=%s',
                        rid, room.player_count)
            emit('error', {'message': '房间未满4人，请先添加AI或等待玩家加入'})
            return
        if not _is_host(room, sid):
            emit('error', {'message': '只有房主可以开始或重新开局'})
            return
        if room.current_game is not None and room.current_game.phase != GamePhase.ROUND_END:
            emit('error', {'message': '当前一局尚未结束'})
            return

        logger.info('WS start_game request: room_id=%s sid=%s', rid, sid)
        restart = bool(room.current_game is not None)
        _do_start_game(rid, room, restart=restart)

    @sio.on('restart_game')
    def on_restart_game(data):
        sid = request.sid
        with session_lock:
            info = player_sessions.get(sid)
        if not info:
            emit('error', {'message': '未加入房间'})
            return
        rid = info['room_id']
        room = room_manager.get(rid)
        if not room:
            emit('error', {'message': '房间不存在'})
            return
        if not room.is_full():
            emit('error', {'message': '房间未满4人，不能重新开始'})
            return
        if not _is_host(room, sid):
            emit('error', {'message': '只有房主可以重新开局'})
            return
        if room.current_game is not None and room.current_game.phase != GamePhase.ROUND_END:
            emit('error', {'message': '当前一局尚未结束'})
            return
        _do_start_game(rid, room, restart=True)

    @sio.on('add_ai')
    def on_add_ai(data):
        sid = request.sid
        with session_lock:
            info = player_sessions.get(sid)
        if not info:
            emit('error', {'message': '未加入房间'})
            return
        rid = info['room_id']
        room = room_manager.get(rid)
        if not room:
            emit('error', {'message': '房间不存在'})
            return
        if room.current_game is not None and room.current_game.phase.value != 'round_end':
            emit('error', {'message': '游戏进行中不能添加AI'})
            return
        if room.is_full():
            emit('error', {'message': '房间已满'})
            return

        ai_model = data.get('model', 'rule')
        logger.info('WS add_ai request: room_id=%s sid=%s model=%s count=%s',
                    rid, sid, ai_model, room.player_count)

        def add_ai_task():
            ok = _add_ai_player(rid, room, ai_model)
            if not ok:
                logger.info('WS add_ai failed: room_id=%s model=%s', rid, ai_model)
                socketio.emit('error', {'message': '添加AI失败'}, to=sid)
                return
            logger.info('WS add_ai success: room_id=%s model=%s count=%s',
                        rid, ai_model, room.player_count)
            socketio.emit('room_state', room.get_room_state(), to=rid)

        threading.Thread(target=add_ai_task, daemon=True).start()

    @sio.on('switch_seat')
    def on_switch_seat(data):
        sid = request.sid
        with session_lock:
            info = player_sessions.get(sid)
        if not info:
            emit('error', {'message': '未加入房间'})
            return
        rid = info['room_id']
        room = room_manager.get(rid)
        if not room:
            emit('error', {'message': '房间不存在'})
            return
        if room.current_game is not None and room.current_game.phase.value != 'round_end':
            emit('error', {'message': '游戏进行中不能换座'})
            return
        try:
            target_seat = int(data.get('seat', -1))
        except (TypeError, ValueError):
            target_seat = -1
        ok, new_seat = room.move_player(sid, target_seat)
        if not ok:
            emit('error', {'message': '换座失败'})
            return
        with session_lock:
            for psid, sinfo in player_sessions.items():
                if sinfo.get('room_id') != rid:
                    continue
                seat_now = room.get_seat_by_id(psid)
                if seat_now >= 0:
                    sinfo['seat'] = seat_now
        logger.info('WS switch_seat success: room_id=%s sid=%s seat=%s', rid, sid, new_seat)
        for p in room.players:
            if p is not None and p.player_id in player_sessions:
                socketio.emit('seat_changed', {'seat': p.seat, 'room_id': rid}, to=p.player_id)
        socketio.emit('room_state', room.get_room_state(), to=rid)

    @sio.on('play_cards')
    def on_play_cards(data):
        sid = request.sid
        with session_lock:
            info = player_sessions.get(sid)
        if not info:
            return
        room = room_manager.get(info['room_id'])
        if not room or not room.current_game:
            return
        indices = data.get('card_indices', [])
        logger.info('WS play_cards request: room_id=%s sid=%s seat=%s indices=%s',
                    info['room_id'], sid, info['seat'], indices)
        result = room.current_game.play_cards(
            info['seat'], indices)
        if not result['success']:
            logger.info('WS play_cards failed: room_id=%s seat=%s message=%s',
                        info['room_id'], info['seat'], result['message'])
            emit('error',
                 {'message': get_msg(result['message'])})
            return
        broadcast_action(info['room_id'], result)
        _handle_post_play(info['room_id'], room, result)

    @sio.on('pass_turn')
    def on_pass(data):
        sid = request.sid
        with session_lock:
            info = player_sessions.get(sid)
        if not info:
            return
        room = room_manager.get(info['room_id'])
        if not room or not room.current_game:
            return
        logger.info('WS pass_turn request: room_id=%s sid=%s seat=%s',
                    info['room_id'], sid, info['seat'])
        result = room.current_game.player_pass(
            info['seat'])
        if not result['success']:
            logger.info('WS pass_turn failed: room_id=%s seat=%s message=%s',
                        info['room_id'], info['seat'], result['message'])
            emit('error',
                 {'message': get_msg(result['message'])})
            return
        broadcast_action(info['room_id'], result)
        broadcast_state(info['room_id'])

    @sio.on('respond_cha')
    def on_respond_cha(data):
        sid = request.sid
        with session_lock:
            info = player_sessions.get(sid)
        if not info:
            return
        room = room_manager.get(info['room_id'])
        if not room or not room.current_game:
            return
        do_cha = data.get('do_cha', False)
        result = room.current_game.respond_cha(
            info['seat'], do_cha)
        if not result['success']:
            emit('error',
                 {'message': get_msg(result['message'])})
            return
        broadcast_action(info['room_id'], result)
        _handle_post_play(info['room_id'], room, result)

    @sio.on('respond_dian')
    def on_respond_dian(data):
        sid = request.sid
        with session_lock:
            info = player_sessions.get(sid)
        if not info:
            return
        room = room_manager.get(info['room_id'])
        if not room or not room.current_game:
            return
        do_dian = data.get('do_dian', False)
        result = room.current_game.respond_dian(
            info['seat'], do_dian)
        if not result['success']:
            emit('error',
                 {'message': get_msg(result['message'])})
            return
        broadcast_action(info['room_id'], result)
        _handle_post_play(info['room_id'], room, result)


def _is_host(room, sid):
    return room.host_player_id == sid


def _do_start_game(room_id, room, restart=False):
    """实际开始游戏或重新发牌"""
    ok = room.start_new_round()
    if not ok:
        logger.info('start game failed: room_id=%s restart=%s', room_id, restart)
        socketio.emit('error', {'message': '无法开始下一局，请确认房间仍有4名玩家'}, to=room_id)
        return
    socketio.emit('room_state', room.get_room_state(), to=room_id)
    broadcast_action(room_id, {
        'action': 'game_started',
        'restart': restart,
        'level_rank': room.get_current_level(),
    })
    broadcast_state(room_id)


def _handle_post_play(room_id, room, result):
    if result.get('game_over'):
        rr = room.process_round_end()
        if rr is not None:
            socketio.emit('room_state', room.get_room_state(), to=room_id)
            socketio.emit('round_end', rr, to=room_id)
    broadcast_state(room_id)
