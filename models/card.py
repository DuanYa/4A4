"""
牌和花色的基础数据模型
"""
from enum import Enum, IntEnum


class Suit(Enum):
    """花色枚举"""
    SPADE = 'spade'      # 黑桃 ♠
    HEART = 'heart'      # 红桃 ♥
    CLUB = 'club'        # 梅花 ♣
    DIAMOND = 'diamond'  # 方块 ♦
    JOKER = 'joker'      # 王牌（无花色）


# 牌面点数到基础权重的映射（不含级牌加成）
BASE_RANK_ORDER = {
    '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12,
    'K': 13, 'A': 14, '2': 15,
    'BJ': 96,  # 小王
    'RJ': 97,  # 大王
}

# 花色显示符号
SUIT_SYMBOLS = {
    Suit.SPADE: '♠',
    Suit.HEART: '♥',
    Suit.CLUB: '♣',
    Suit.DIAMOND: '♦',
    Suit.JOKER: '🃏',
}

# 所有常规点数（不含王）
REGULAR_RANKS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']

# 可以组成顺子的点数序列（不含2和王）
STRAIGHT_RANKS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']


class Card:
    """单张牌"""

    def __init__(self, suit: Suit, rank: str):
        self.suit = suit
        self.rank = rank

    def get_value(self, level_rank: str) -> int:
        """
        获取牌的权重值
        level_rank: 当前级牌的点数（如 '3' 表示打3）
        级牌权重设为98，介于小王(96)和大王(97)之间的上方... 
        实际上级牌 > 2(15) 且 < 小王(96)，设为16即可保证大于2
        但按规则：大王 > 小王 > 级牌 > 2，所以级牌权重设为16
        """
        if self.rank == level_rank and self.suit != Suit.JOKER:
            return 16  # 级牌权重：大于2(15)，小于小王(96)
        return BASE_RANK_ORDER.get(self.rank, 0)

    def is_joker(self) -> bool:
        return self.suit == Suit.JOKER

    def is_red_joker(self) -> bool:
        return self.rank == 'RJ'

    def is_black_joker(self) -> bool:
        return self.rank == 'BJ'

    def to_dict(self) -> dict:
        return {
            'suit': self.suit.value,
            'rank': self.rank,
            'display': str(self),
        }

    def __str__(self):
        if self.is_red_joker():
            return '大王'
        if self.is_black_joker():
            return '小王'
        return f"{SUIT_SYMBOLS[self.suit]}{self.rank}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank

    def __hash__(self):
        return hash((self.suit, self.rank))


def create_full_deck() -> list:
    """创建一副完整的54张扑克牌"""
    deck = []
    for suit in [Suit.SPADE, Suit.HEART, Suit.CLUB, Suit.DIAMOND]:
        for rank in REGULAR_RANKS:
            deck.append(Card(suit, rank))
    deck.append(Card(Suit.JOKER, 'BJ'))
    deck.append(Card(Suit.JOKER, 'RJ'))
    return deck


def sort_cards(cards: list, level_rank: str) -> list:
    """按权重从大到小排序手牌"""
    return sorted(cards, key=lambda c: (-c.get_value(level_rank), c.suit.value))
