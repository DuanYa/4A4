"""
房间管理与升级系统
"""
import threading
from models.player import Player
from models.game import Game, GamePhase

# 级别顺序
LEVEL_ORDER = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
# 必须全洞才能升级的级别
MUST_QUAN_DONG = {'3', 'J', 'A'}


class Room:
    """游戏房间 - 管理多局游戏和升级系统"""

    def __init__(self, room_id):
        self.room_id = room_id
        self.lock = threading.Lock()
        self.players = [None, None, None, None]
        self.player_count = 0
        # 升级系统
        self.team_levels = {0: '3', 1: '3'}  # 各队当前级别
        self.on_stage_team = 0  # 台上队伍
        self.current_game = None
        self.round_history = []
        self.host_player_id = None

    def add_player(self, player_id, name, is_ai=False):
        """添加玩家到房间，返回座位号或-1"""
        with self.lock:
            for i in range(4):
                if self.players[i] is None:
                    player = Player(player_id, name, i)
                    player.is_ai = is_ai
                    self.players[i] = player
                    if self.host_player_id is None and not is_ai:
                        self.host_player_id = player_id
                    self.player_count += 1
                    return i
            return -1

    def remove_player(self, player_id):
        with self.lock:
            for i in range(4):
                if (self.players[i] is not None
                        and self.players[i].player_id == player_id):
                    self.players[i] = None
                    self.player_count -= 1
                    if self.host_player_id == player_id:
                        self.host_player_id = None
                        for np in self.players:
                            if np is not None and not getattr(np, 'is_ai', False):
                                self.host_player_id = np.player_id
                                break
                    return i
            return -1

    def get_player_by_id(self, player_id):
        for p in self.players:
            if p is not None and p.player_id == player_id:
                return p
        return None

    def get_seat_by_id(self, player_id):
        for p in self.players:
            if p is not None and p.player_id == player_id:
                return p.seat
        return -1

    def move_player(self, player_id, target_seat):
        """移动玩家到目标座位；目标为空则移动，目标有人则交换。"""
        with self.lock:
            if target_seat < 0 or target_seat >= 4:
                return False, -1
            source_seat = -1
            for i, p in enumerate(self.players):
                if p is not None and p.player_id == player_id:
                    source_seat = i
                    break
            if source_seat < 0:
                return False, -1
            if source_seat == target_seat:
                return True, source_seat
            source_player = self.players[source_seat]
            target_player = self.players[target_seat]
            self.players[source_seat], self.players[target_seat] = target_player, source_player
            source_player.seat = target_seat
            if target_player is not None:
                target_player.seat = source_seat
            return True, target_seat

    def is_full(self):
        return self.player_count >= 4

    def get_current_level(self):
        """获取当前台上队伍的级牌"""
        return self.team_levels[self.on_stage_team]

    def start_new_round(self):
        """开始新一局"""
        with self.lock:
            if not self.is_full():
                return False
            if self.current_game is not None and self.current_game.phase != GamePhase.ROUND_END:
                return False
            level = self.get_current_level()
            self.current_game = Game(
                list(self.players), level, self.on_stage_team
            )
            self.current_game.start()
            return True

    def process_round_end(self):
        """处理一局结束后的升级逻辑"""
        if self.current_game is None:
            return None
        if self.current_game.phase != GamePhase.ROUND_END:
            return None

        rr = self.current_game.get_round_result()
        self.round_history.append(rr)

        result = dict(rr)
        winner_team = -1
        upgrade = 0

        for team in [0, 1]:
            tr = rr.get('team%d_result' % team)
            if tr == 'quan_dong':
                winner_team = team
                upgrade = 2
            elif tr == 'ban_dong':
                winner_team = team
                upgrade = 1

        result['winner_team'] = winner_team
        result['upgrade'] = 0
        result['new_level'] = self.team_levels.copy()
        result['stage_change'] = False

        if winner_team == -1:
            return result

        current_level = self.team_levels[winner_team]
        # 特殊级别需要全洞
        if current_level in MUST_QUAN_DONG and upgrade < 2:
            upgrade = 0

        if upgrade > 0:
            new_level = self._advance_level(
                current_level, upgrade)
            self.team_levels[winner_team] = new_level
            result['upgrade'] = upgrade
            result['new_level'] = self.team_levels.copy()

        # 台上/台下切换
        if winner_team != self.on_stage_team:
            # 直J/直A检测
            old_level = self.team_levels[self.on_stage_team]
            if old_level == 'J':
                self.team_levels[self.on_stage_team] = '3'
                result['zhi_j'] = True
            elif old_level == 'A':
                self.team_levels[self.on_stage_team] = 'J'
                result['zhi_a'] = True

            self.on_stage_team = winner_team
            result['stage_change'] = True

        result['on_stage_team'] = self.on_stage_team
        return result

    def _advance_level(self, current, steps):
        """前进级别"""
        idx = LEVEL_ORDER.index(current)
        new_idx = min(idx + steps, len(LEVEL_ORDER) - 1)
        return LEVEL_ORDER[new_idx]

    def get_room_state(self):
        players_data = []
        for p in self.players:
            if p is not None:
                players_data.append({
                    'seat': p.seat, 'name': p.name,
                    'player_id': p.player_id, 'team': p.team,
                    'is_ai': getattr(p, 'is_ai', False),
                })
            else:
                players_data.append(None)
        return {
            'room_id': self.room_id,
            'player_count': self.player_count,
            'host_player_id': self.host_player_id,
            'players': players_data,
            'team_levels': self.team_levels,
            'on_stage_team': self.on_stage_team,
            'game_active': (
                self.current_game is not None
                and self.current_game.phase != GamePhase.ROUND_END),
        }
