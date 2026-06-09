"""
玩家模型
"""
from models.card import Card, sort_cards


class Player:
    """玩家"""

    def __init__(self, player_id: str, name: str, seat: int, user_id=None,
                 avatar_url=''):
        """
        player_id: 唯一标识（对应websocket的sid）
        name: 显示名称
        seat: 座位号 0-3
        """
        self.player_id = player_id
        self.user_id = user_id
        self.name = name
        self.avatar_url = avatar_url
        self.seat = seat
        self.online = True
        self.hand = []          # 手牌列表
        self.finished = False   # 是否已出完牌
        self.finish_order = -1  # 出完牌的顺序(0=第一个出完)

    @property
    def team(self) -> int:
        """队伍编号：0=A队(座位0,2)，1=B队(座位1,3)"""
        return self.seat % 2

    def set_hand(self, cards: list):
        self.hand = list(cards)

    def remove_cards(self, cards: list):
        """从手牌中移除指定的牌"""
        for card in cards:
            self.hand.remove(card)

    def has_cards(self, cards: list) -> bool:
        """检查手牌中是否包含指定的牌"""
        hand_copy = list(self.hand)
        for card in cards:
            if card in hand_copy:
                hand_copy.remove(card)
            else:
                return False
        return True

    def has_rank_pair(self, rank: str) -> bool:
        """检查手牌中是否有指定点数的对子（用于叉牌判断）"""
        count = sum(1 for c in self.hand if c.rank == rank and not c.is_joker())
        return count >= 2

    def has_rank_single(self, rank: str) -> bool:
        """检查手牌中是否有指定点数的单张（用于点牌判断）"""
        return any(c.rank == rank and not c.is_joker() for c in self.hand)

    def get_cards_by_rank(self, rank: str, count: int) -> list:
        """获取手牌中指定点数的牌"""
        result = []
        for c in self.hand:
            if c.rank == rank and not c.is_joker() and len(result) < count:
                result.append(c)
        return result

    def sort_hand(self, level_rank: str):
        self.hand = sort_cards(self.hand, level_rank)

    def hand_size(self) -> int:
        return len(self.hand)

    def to_dict(self, hide_hand: bool = False) -> dict:
        return {
            'player_id': self.player_id,
            'user_id': self.user_id,
            'name': self.name,
            'avatar_url': self.avatar_url,
            'seat': self.seat,
            'team': self.team,
            'online': self.online,
            'hand_size': self.hand_size(),
            'hand': [] if hide_hand else [c.to_dict() for c in self.hand],
            'finished': self.finished,
            'finish_order': self.finish_order,
        }
