"""
AI玩家 - 通过WebSocket客户端连接服务器，模拟真实玩家
AI理解游戏规则，能正确出牌、pass、叉牌和点牌
"""
import threading
import time
import logging
import socketio
from models.card import Card, Suit, STRAIGHT_RANKS
from models.hand_type import (
    identify_hand, can_beat, HandType, HandCategory
)


def card_from_dict(d):
    """从服务端字典数据还原Card对象"""
    suit = Suit(d['suit'])
    return Card(suit, d['rank'])


class AIPlayer:
    """AI玩家WebSocket客户端"""

    def __init__(self, name, server_url='http://localhost:5000',
                 model_name='rule'):
        self.name = name
        self.server_url = server_url
        self.model_name = model_name or 'rule'
        self.policy = None
        self.sio = socketio.Client()
        self.seat = -1
        self.room_id = ''
        self.hand = []
        self.game_state = None
        self.level_rank = '3'
        self.connected = False
        self.lock = threading.Lock()
        self._setup_handlers()

    def _setup_handlers(self):
        @self.sio.on('connect')
        def on_connect():
            self.connected = True

        @self.sio.on('disconnect')
        def on_disconnect():
            self.connected = False

        @self.sio.on('joined')
        def on_joined(data):
            self.seat = data['seat']
            self.room_id = data['room_id']

        @self.sio.on('error')
        def on_error(data):
            pass

        @self.sio.on('room_state')
        def on_room_state(data):
            pass

        @self.sio.on('game_state')
        def on_game_state(data):
            self._handle_game_state(data)

        @self.sio.on('game_action')
        def on_game_action(data):
            if data.get('action') == 'game_started':
                self.level_rank = data.get('level_rank', '3')

        @self.sio.on('round_end')
        def on_round_end(data):
            pass

    def connect_and_join(self, room_id):
        """连接服务器并加入房间"""
        self.room_id = room_id
        self.sio.connect(
            self.server_url, transports=['websocket'])
        for _ in range(50):
            if self.connected:
                break
            time.sleep(0.05)
        self.sio.emit('join_room', {
            'room_id': room_id, 'name': self.name, 'is_ai': True})
        for _ in range(50):
            if self.seat >= 0:
                break
            time.sleep(0.05)

    def disconnect(self):
        if self.connected:
            try:
                self.sio.disconnect()
            except Exception:
                pass
        self.connected = False

    def _handle_game_state(self, state):
        """处理游戏状态，决定是否行动"""
        with self.lock:
            self.game_state = state
            self.level_rank = state.get('level_rank', '3')

            for p in state['players']:
                if p['seat'] == self.seat:
                    self.hand = [card_from_dict(c)
                                 for c in p.get('hand', [])]
                    break

            phase = state['phase']
            delay = 0.5

            if (phase == 'cha_asking'
                    and state.get('cha_asking_seat') == self.seat):
                threading.Timer(
                    delay, self._decide_cha).start()
                return

            if (phase == 'dian_asking'
                    and state.get('dian_asking_seat') == self.seat):
                threading.Timer(
                    delay, self._decide_dian).start()
                return

            if (phase == 'playing'
                    and state.get('current_player_seat') == self.seat):
                threading.Timer(
                    delay, self._decide_play).start()

    def _decide_play(self):
        try:
            self._do_decide_play()
        except Exception:
            import traceback; traceback.print_exc()

    def _do_decide_play(self):
        if not self.connected:
            return
        state = self.game_state
        if state is None:
            return
        is_free = state.get('is_free_play', False)
        last_ht_dict = state.get('last_hand_type')

        if is_free:
            play = self._choose_play_with_model(None, is_free)
            if play is not None:
                self.sio.emit('play_cards',
                              {'card_indices': play})
            return

        if last_ht_dict is None:
            self.sio.emit('pass_turn', {})
            return

        last_ht = _reconstruct_ht(last_ht_dict)
        play = self._choose_play_with_model(last_ht, is_free)
        if play is not None:
            self.sio.emit('play_cards',
                          {'card_indices': play})
        else:
            self.sio.emit('pass_turn', {})

    def _choose_play_with_model(self, last_ht, is_free):
        """根据配置选择规则AI或深度学习AI"""
        if self.model_name in ('rule', 'heuristic', ''):
            if is_free:
                return self._find_free_play()
            return self._find_beat_play(last_ht)

        def fallback():
            if is_free:
                return self._find_free_play()
            return self._find_beat_play(last_ht)

        try:
            if self.policy is None:
                from rl.policy import NeuralPolicy
                self.policy = NeuralPolicy(self.model_name)
            return self.policy.choose_play(
                self.game_state, self.hand, self.level_rank,
                self.seat, last_ht, is_free, fallback)
        except Exception:
            return fallback()

    def _decide_cha(self):
        try:
            if self.connected:
                self.sio.emit('respond_cha', {
                    'do_cha': self._should_cha()})
        except Exception:
            import traceback; traceback.print_exc()

    def _decide_dian(self):
        try:
            if self.connected:
                self.sio.emit('respond_dian', {
                    'do_dian': self._should_dian()})
        except Exception:
            import traceback; traceback.print_exc()

    def _player_hand_size(self, seat):
        state = self.game_state or {}
        for player in state.get('players', []):
            if player.get('seat') == seat:
                return player.get('hand_size', 14)
        return 14

    def _same_team(self, seat):
        return self.seat >= 0 and seat >= 0 and seat % 2 == self.seat % 2

    def _should_cha(self):
        """Avoid stealing tempo from a short-card teammate."""
        state = self.game_state or {}
        source = state.get('cha_source_seat', -1)
        own_after = max(0, len(self.hand) - 2)
        if own_after <= 2:
            return True
        if self._same_team(source) and self._player_hand_size(source) <= 2:
            return False
        return True

    def _should_dian(self):
        """Point opponents freely, but do not undercut a short teammate's cha."""
        state = self.game_state or {}
        cha_player = state.get('cha_player_seat', -1)
        own_after = max(0, len(self.hand) - 1)
        if own_after <= 2:
            return True
        if self._same_team(cha_player) and self._player_hand_size(cha_player) <= 2:
            return False
        return True


# Mixin: inject search methods from ai_search
from models.ai_search import _reconstruct_ht
from models import ai_search as _ais
AIPlayer._find_free_play = _ais._find_free_play
AIPlayer._find_beat_play = _ais._find_beat_play
