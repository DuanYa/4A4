"""
单局游戏逻辑 - 核心类定义与初始化、出牌逻辑
"""
import threading
import logging
from enum import Enum
from models.card import Card, Suit
from models.player import Player
from models.deck import shuffle_and_deal
from models.hand_type import (
    HandType, HandCategory, identify_hand, can_beat
)

logger = logging.getLogger('game')


class GamePhase(Enum):
    WAITING = 'waiting'
    DEALING = 'dealing'
    TRIBUTE = 'tribute'
    PLAYING = 'playing'
    CHA_ASKING = 'cha_asking'
    DIAN_ASKING = 'dian_asking'
    ROUND_END = 'round_end'


class Game:
    def __init__(self, players, level_rank, on_stage_team):
        self.players = players
        self.level_rank = level_rank
        self.on_stage_team = on_stage_team
        self.phase = GamePhase.WAITING
        self.lock = threading.Lock()
        self.current_player_seat = -1
        self.last_play_seat = -1
        self.last_hand_type = None
        self.pass_count = 0
        self.is_free_play = True
        self.played_cards = []
        self.cha_rank = None
        self.cha_source_seat = -1
        self.cha_player_seat = -1
        self.cha_asking_seat = -1
        self.cha_candidates = []
        self.dian_asking_seat = -1
        self.dian_candidates = []
        self.is_cha_active = False
        self.finish_order = []
        self.finished_count = 0
        self.play_history = []

    def start(self):
        with self.lock:
            self.phase = GamePhase.DEALING
            hands = shuffle_and_deal(self.level_rank)
            for i, p in enumerate(self.players):
                p.set_hand(hands[i])
                p.finished = False
                p.finish_order = -1
            self.finish_order = []
            self.finished_count = 0
            self.play_history = []
            self.current_player_seat = self._find_heart_3()
            self.is_free_play = True
            self.last_hand_type = None
            self.last_play_seat = -1
            self.pass_count = 0
            self.phase = GamePhase.PLAYING
            logger.info('Game start: level=%s on_stage_team=%s first_seat=%s hand_sizes=%s',
                        self.level_rank, self.on_stage_team,
                        self.current_player_seat,
                        [p.hand_size() for p in self.players])

    def _find_heart_3(self):
        for p in self.players:
            for c in p.hand:
                if c.suit == Suit.HEART and c.rank == '3':
                    return p.seat
        return 0

    def get_active_seats(self):
        return [p.seat for p in self.players if not p.finished]

    def _next_active_seat(self, cur):
        active = self.get_active_seats()
        if not active:
            return -1
        for i in range(1, 5):
            ns = (cur + i) % 4
            if ns in active:
                return ns
        return -1

    def _mark_finished(self, player, result):
        player.finished = True
        player.finish_order = self.finished_count
        self.finish_order.append(player.seat)
        self.finished_count += 1
        result['player_finished'] = True

    def _check_game_over(self, result):
        finished_by_team = {0: [], 1: []}
        for p in self.players:
            if p.finished:
                finished_by_team[p.team].append(p)
        winner_team = None
        for team, members in finished_by_team.items():
            if len(members) >= 2:
                winner_team = team
                break
        if winner_team is None and self.finished_count >= 3:
            for p in self.players:
                if p.finished:
                    winner_team = p.team
                    break
        if winner_team is None:
            return False
        for p in self.players:
            if not p.finished:
                p.finished = True
                p.finish_order = self.finished_count
                self.finish_order.append(p.seat)
                self.finished_count += 1
        self.phase = GamePhase.ROUND_END
        result['game_over'] = True
        result['winner_team'] = winner_team
        result['finish_order'] = list(self.finish_order)
        return True

    def play_cards(self, seat, card_indices):
        with self.lock:
            return self._play_cards_inner(seat, card_indices)

    def _play_cards_inner(self, seat, card_indices):
        if self.phase != GamePhase.PLAYING:
            return {'success': False, 'message': 'not_playing'}
        if seat != self.current_player_seat:
            return {'success': False, 'message': 'not_your_turn'}
        player = self.players[seat]
        if player.finished:
            return {'success': False, 'message': 'already_finished'}
        if not card_indices:
            return {'success': False, 'message': 'no_cards_selected'}
        if any(i < 0 or i >= len(player.hand) for i in card_indices):
            return {'success': False, 'message': 'invalid_index'}
        indices = sorted(set(card_indices))
        cards = [player.hand[i] for i in indices]
        hand_type = identify_hand(cards, self.level_rank)
        logger.info('Game play attempt: seat=%s indices=%s cards=%s hand_type=%s free=%s last=%s',
                    seat, indices, [str(c) for c in cards],
                    hand_type.category.name if hand_type else None,
                    self.is_free_play,
                    self.last_hand_type.category.name if self.last_hand_type else None)
        if hand_type is None:
            return {'success': False, 'message': 'invalid_hand'}
        if self.is_free_play:
            if hand_type.category == HandCategory.DOUBLE_STRAIGHT:
                if len(cards) != len(player.hand):
                    return {'success': False, 'message': 'no_double_straight_lead'}
        if not self.is_free_play:
            if not can_beat(self.last_hand_type, hand_type):
                return {'success': False, 'message': 'cannot_beat'}
        player.remove_cards(cards)
        logger.info('Game play success: seat=%s hand_type=%s cards=%s remaining=%s',
                    seat, hand_type.category.name, [str(c) for c in cards],
                    player.hand_size())
        self.play_history.append({
            'seat': seat,
            'action': 'play',
            'cards': [c.to_dict() for c in cards],
            'hand_type': hand_type.to_dict(),
            'hand_size_after': player.hand_size(),
        })
        self.last_hand_type = hand_type
        self.last_play_seat = seat
        self.pass_count = 0
        self.is_free_play = False
        self.played_cards = cards
        result = {
            'success': True, 'seat': seat,
            'cards': [c.to_dict() for c in cards],
            'hand_type': hand_type.to_dict(),
        }
        if player.hand_size() == 0:
            self._mark_finished(player, result)
        if self._check_game_over(result):
            return result
        if hand_type.category == HandCategory.SINGLE:
            if not cards[0].is_joker():
                if self._check_cha(seat, cards[0].rank):
                    result['cha_checking'] = True
                    return result
        self._advance_to_next(seat)
        result['next_seat'] = self.current_player_seat
        logger.info('Game advance: from=%s to=%s free=%s last_play=%s pass_count=%s',
                    seat, self.current_player_seat, self.is_free_play,
                    self.last_play_seat, self.pass_count)
        return result

    def _advance_to_next(self, cur_seat):
        ns = self._next_active_seat(cur_seat)
        if not self.is_free_play and ns == self.last_play_seat:
            lp = self.players[self.last_play_seat]
            if lp.finished:
                tm = (self.last_play_seat + 2) % 4
                if self.players[tm].finished:
                    self.current_player_seat = (
                        self._next_active_seat(self.last_play_seat))
                else:
                    self.current_player_seat = tm
            else:
                self.current_player_seat = self.last_play_seat
            self.is_free_play = True
            self.last_hand_type = None
            self.pass_count = 0
        else:
            self.current_player_seat = ns


# Mixin: inject cha/dian/pass/state methods from game_actions
from models import game_actions as _ga
Game._set_free_play_after_finished_or_self = _ga._set_free_play_after_finished_or_self
Game._check_cha = _ga._check_cha
Game.respond_cha = _ga.respond_cha
Game._respond_cha_inner = _ga._respond_cha_inner
Game._check_dian = _ga._check_dian
Game.respond_dian = _ga.respond_dian
Game._respond_dian_inner = _ga._respond_dian_inner
Game.player_pass = _ga.player_pass
Game._player_pass_inner = _ga._player_pass_inner
Game.get_state = _ga.get_state
Game.get_round_result = _ga.get_round_result
