"""
叉/点/pass/状态查询逻辑 - Game类的扩展方法
通过 game.py 末尾的 mixin 导入方式注入
"""
from models.hand_type import identify_hand
import logging

logger = logging.getLogger('game.actions')


def _set_free_play_after_finished_or_self(self, seat):
    player = self.players[seat]
    if not player.finished:
        self.current_player_seat = seat
        return
    teammate = (seat + 2) % 4
    if not self.players[teammate].finished:
        self.current_player_seat = teammate
        return
    self.current_player_seat = self._next_active_seat(seat)


def _check_cha(self, source_seat, rank):
    """检查是否有人可以叉牌"""
    self.cha_rank = rank
    self.cha_source_seat = source_seat
    self.cha_candidates = []
    for i in range(1, 4):
        cs = (source_seat + i) % 4
        p = self.players[cs]
        if not p.finished and p.has_rank_pair(rank):
            self.cha_candidates.append(cs)
    if not self.cha_candidates:
        logger.info('Cha check: source=%s rank=%s candidates=[]', source_seat, rank)
        return False
    logger.info('Cha check: source=%s rank=%s candidates=%s',
                source_seat, rank, self.cha_candidates)
    from models.game import GamePhase
    self.phase = GamePhase.CHA_ASKING
    self.cha_asking_seat = self.cha_candidates[0]
    return True


def respond_cha(self, seat, do_cha):
    with self.lock:
        return self._respond_cha_inner(seat, do_cha)


def _respond_cha_inner(self, seat, do_cha):
    from models.game import GamePhase
    if self.phase != GamePhase.CHA_ASKING:
        return {'success': False, 'message': 'not_cha_phase'}
    if seat != self.cha_asking_seat:
        return {'success': False, 'message': 'not_asked'}
    if do_cha:
        logger.info('Cha response: seat=%s do_cha=True rank=%s', seat, self.cha_rank)
        player = self.players[seat]
        cha_cards = player.get_cards_by_rank(self.cha_rank, 2)
        if len(cha_cards) < 2:
            return {'success': False, 'message': 'not_enough_cards'}
        player.remove_cards(cha_cards)
        self.play_history.append({
            'seat': seat,
            'action': 'cha',
            'cards': [c.to_dict() for c in cha_cards],
            'hand_type': identify_hand(cha_cards, self.level_rank).to_dict(),
            'hand_size_after': player.hand_size(),
        })
        cha_hand = identify_hand(cha_cards, self.level_rank)
        self.last_hand_type = cha_hand
        self.last_play_seat = seat
        self.cha_player_seat = seat
        self.is_cha_active = True
        self.played_cards = cha_cards
        result = {
            'success': True, 'action': 'cha', 'seat': seat,
            'cards': [c.to_dict() for c in cha_cards],
        }
        if player.hand_size() == 0:
            self._mark_finished(player, result)
        if self._check_game_over(result):
            return result
        if self._check_dian(seat, self.cha_rank):
            result['dian_checking'] = True
        else:
            self.phase = GamePhase.PLAYING
            self.is_cha_active = False
            self._set_free_play_after_finished_or_self(seat)
            self.is_free_play = True
            self.last_hand_type = None
            self.pass_count = 0
            result['next_seat'] = self.current_player_seat
        return result
    else:
        logger.info('Cha response: seat=%s do_cha=False', seat)
        idx = self.cha_candidates.index(seat)
        if idx + 1 < len(self.cha_candidates):
            self.cha_asking_seat = self.cha_candidates[idx + 1]
            return {
                'success': True, 'action': 'cha_pass',
                'seat': seat,
                'next_ask_seat': self.cha_asking_seat,
            }
        else:
            self.phase = GamePhase.PLAYING
            self._advance_to_next(self.cha_source_seat)
            return {
                'success': True, 'action': 'cha_all_pass',
                'next_seat': self.current_player_seat,
            }


def _check_dian(self, cha_seat, rank):
    """检查是否有人可以点牌"""
    self.dian_candidates = []
    for i in range(1, 4):
        cs = (cha_seat + i) % 4
        p = self.players[cs]
        if not p.finished and p.has_rank_single(rank):
            self.dian_candidates.append(cs)
    if not self.dian_candidates:
        logger.info('Dian check: cha_seat=%s rank=%s candidates=[]', cha_seat, rank)
        return False
    logger.info('Dian check: cha_seat=%s rank=%s candidates=%s',
                cha_seat, rank, self.dian_candidates)
    from models.game import GamePhase
    self.phase = GamePhase.DIAN_ASKING
    self.dian_asking_seat = self.dian_candidates[0]
    return True


def respond_dian(self, seat, do_dian):
    with self.lock:
        return self._respond_dian_inner(seat, do_dian)


def _respond_dian_inner(self, seat, do_dian):
    from models.game import GamePhase
    if self.phase != GamePhase.DIAN_ASKING:
        return {'success': False, 'message': 'not_dian_phase'}
    if seat != self.dian_asking_seat:
        return {'success': False, 'message': 'not_asked'}
    if do_dian:
        logger.info('Dian response: seat=%s do_dian=True rank=%s', seat, self.cha_rank)
        player = self.players[seat]
        dian_cards = player.get_cards_by_rank(self.cha_rank, 1)
        if not dian_cards:
            return {'success': False, 'message': 'no_card_to_dian'}
        player.remove_cards(dian_cards)
        self.play_history.append({
            'seat': seat,
            'action': 'dian',
            'cards': [c.to_dict() for c in dian_cards],
            'hand_type': identify_hand(dian_cards, self.level_rank).to_dict(),
            'hand_size_after': player.hand_size(),
        })
        self.played_cards = dian_cards
        result = {
            'success': True, 'action': 'dian', 'seat': seat,
            'cards': [c.to_dict() for c in dian_cards],
        }
        if player.hand_size() == 0:
            self._mark_finished(player, result)
        if self._check_game_over(result):
            return result
        self.phase = GamePhase.PLAYING
        self.is_cha_active = False
        self._set_free_play_after_finished_or_self(seat)
        self.is_free_play = True
        self.last_hand_type = None
        self.pass_count = 0
        result['next_seat'] = self.current_player_seat
        return result
    else:
        logger.info('Dian response: seat=%s do_dian=False', seat)
        idx = self.dian_candidates.index(seat)
        if idx + 1 < len(self.dian_candidates):
            self.dian_asking_seat = self.dian_candidates[idx + 1]
            return {
                'success': True, 'action': 'dian_pass',
                'seat': seat,
                'next_ask_seat': self.dian_asking_seat,
            }
        else:
            self.phase = GamePhase.PLAYING
            self.is_cha_active = False
            s = self.cha_player_seat
            self._set_free_play_after_finished_or_self(s)
            self.is_free_play = True
            self.last_hand_type = None
            self.pass_count = 0
            return {
                'success': True, 'action': 'dian_all_pass',
                'next_seat': self.current_player_seat,
            }


def player_pass(self, seat):
    with self.lock:
        return self._player_pass_inner(seat)


def _player_pass_inner(self, seat):
    from models.game import GamePhase
    if self.phase != GamePhase.PLAYING:
        return {'success': False, 'message': 'not_playing'}
    if seat != self.current_player_seat:
        return {'success': False, 'message': 'not_your_turn'}
    if self.is_free_play:
        return {'success': False, 'message': 'must_play_as_leader'}
    self.pass_count += 1
    logger.info('Pass: seat=%s pass_count=%s last_play=%s free=%s',
                seat, self.pass_count, self.last_play_seat, self.is_free_play)
    self.play_history.append({
        'seat': seat,
        'action': 'pass',
        'cards': [],
        'hand_type': None,
        'hand_size_after': self.players[seat].hand_size(),
    })
    active_others = [
        s for s in self.get_active_seats()
        if s != self.last_play_seat
    ]
    if self.pass_count >= len(active_others):
        last_seat = self.last_play_seat
        lp = self.players[last_seat]
        if lp.finished:
            tm = (last_seat + 2) % 4
            if self.players[tm].finished:
                self.current_player_seat = (
                    self._next_active_seat(last_seat))
            else:
                self.current_player_seat = tm
        else:
            self.current_player_seat = last_seat
        self.is_free_play = True
        self.last_hand_type = None
        self.pass_count = 0
        return {
            'success': True, 'action': 'pass', 'seat': seat,
            'new_round': True,
            'next_seat': self.current_player_seat,
        }
    self._advance_to_next(seat)
    return {
        'success': True, 'action': 'pass', 'seat': seat,
        'next_seat': self.current_player_seat,
    }


def get_state(self, for_seat=-1):
    from models.game import GamePhase
    players_data = []
    for p in self.players:
        hide = for_seat != -1 and p.seat != for_seat
        players_data.append(p.to_dict(hide_hand=hide))
    return {
        'phase': self.phase.value,
        'level_rank': self.level_rank,
        'on_stage_team': self.on_stage_team,
        'current_player_seat': self.current_player_seat,
        'last_play_seat': self.last_play_seat,
        'last_hand_type': (
            self.last_hand_type.to_dict()
            if self.last_hand_type else None),
        'is_free_play': self.is_free_play,
        'played_cards': [c.to_dict() for c in self.played_cards],
        'players': players_data,
        'play_history': list(self.play_history[-80:]),
        'finish_order': self.finish_order,
        'cha_asking_seat': (
            self.cha_asking_seat
            if self.phase == GamePhase.CHA_ASKING else -1),
        'dian_asking_seat': (
            self.dian_asking_seat
            if self.phase == GamePhase.DIAN_ASKING else -1),
        'cha_rank': self.cha_rank,
        'cha_source_seat': self.cha_source_seat,
        'cha_player_seat': self.cha_player_seat,
    }


def get_round_result(self):
    from models.game import GamePhase
    if self.phase != GamePhase.ROUND_END:
        return None
    t0 = sorted(
        [p.finish_order for p in self.players if p.team == 0])
    t1 = sorted(
        [p.finish_order for p in self.players if p.team == 1])
    result = {
        'finish_order': self.finish_order,
        'team0_orders': t0, 'team1_orders': t1,
    }
    for team, orders in [(0, t0), (1, t1)]:
        if orders == [0, 1]:
            result['team%d_result' % team] = 'quan_dong'
        elif orders == [0, 2]:
            result['team%d_result' % team] = 'ban_dong'
        else:
            result['team%d_result' % team] = 'lose'
    return result
