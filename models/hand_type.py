"""
牌型识别与比较
牌型: 单张, 对子, 单龙(顺子), 双龙(连对), 炸(三条), 轰(四条), 双王, 四幺四
"""
from enum import IntEnum
from models.card import Card, Suit, STRAIGHT_RANKS, BASE_RANK_ORDER


class HandCategory(IntEnum):
    """牌型类别，数值越大越强"""
    SINGLE = 1       # 单张
    PAIR = 2         # 对子
    STRAIGHT = 3     # 单龙（顺子）
    DOUBLE_STRAIGHT = 4  # 双龙（连对）
    BOMB3 = 5        # 炸（三条）
    BOMB4 = 6        # 轰（四条）
    JOKER_BOMB = 7   # 双王
    SI_YAO_SI = 8    # 四幺四（最大）


class HandType:
    """牌型描述"""

    def __init__(self, category: HandCategory, cards: list, key_value: int = 0, length: int = 0):
        """
        category: 牌型类别
        cards: 组成牌型的牌列表
        key_value: 用于比较大小的关键值（牌点权重）
        length: 龙的长度（仅对单龙/双龙有意义）
        """
        self.category = category
        self.cards = list(cards)
        self.key_value = key_value
        self.length = length

    def to_dict(self) -> dict:
        return {
            'category': self.category.name,
            'cards': [c.to_dict() for c in self.cards],
            'key_value': self.key_value,
            'length': self.length,
        }

    def __repr__(self):
        return f"HandType({self.category.name}, {self.cards}, key={self.key_value}, len={self.length})"


def _get_straight_index(rank: str) -> int:
    """获取点数在顺子序列中的索引，不在序列中返回-1"""
    if rank in STRAIGHT_RANKS:
        return STRAIGHT_RANKS.index(rank)
    return -1


def identify_hand(cards: list, level_rank: str) -> HandType:
    """
    识别一组牌的牌型
    返回 HandType 或 None（非法牌型）
    """
    if not cards:
        return None

    n = len(cards)

    # === 单张 ===
    if n == 1:
        card = cards[0]
        return HandType(HandCategory.SINGLE, cards, card.get_value(level_rank))

    # === 两张牌的情况 ===
    if n == 2:
        c1, c2 = cards[0], cards[1]

        # 双王
        if {c1.rank, c2.rank} == {'BJ', 'RJ'}:
            return HandType(HandCategory.JOKER_BOMB, cards, 200)

        # 对子：两张牌点相同（花色不同）
        if c1.rank == c2.rank and c1.suit != Suit.JOKER and c2.suit != Suit.JOKER:
            return HandType(HandCategory.PAIR, cards, c1.get_value(level_rank))

        return None

    # === 三张牌的情况 ===
    if n == 3:
        ranks = [c.rank for c in cards]

        # 四幺四：两张4 + 一张A
        if sorted(ranks) == ['4', '4', 'A']:
            non_joker = all(c.suit != Suit.JOKER for c in cards)
            if non_joker:
                return HandType(HandCategory.SI_YAO_SI, cards, 300)

        # 炸（三条）：三张牌点相同
        if len(set(ranks)) == 1 and all(c.suit != Suit.JOKER for c in cards):
            return HandType(HandCategory.BOMB3, cards, cards[0].get_value(level_rank))

        # 三张单龙
        return _try_straight(cards, level_rank)

    # === 四张牌 ===
    if n == 4:
        ranks = [c.rank for c in cards]

        # 轰（四条）：四张牌点相同
        if len(set(ranks)) == 1 and all(c.suit != Suit.JOKER for c in cards):
            return HandType(HandCategory.BOMB4, cards, cards[0].get_value(level_rank))

        # 四张单龙
        return _try_straight(cards, level_rank)

    # === 5张及以上 ===
    # 尝试双龙（连对）
    dbl = _try_double_straight(cards, level_rank)
    if dbl:
        return dbl

    # 尝试单龙（顺子）
    return _try_straight(cards, level_rank)


def _try_straight(cards: list, level_rank: str) -> HandType:
    """尝试识别单龙（顺子）"""
    n = len(cards)
    if n < 3:
        return None

    # 不能包含2、王、级牌
    for c in cards:
        if c.rank == '2' or c.is_joker() or c.rank == level_rank:
            return None

    # 获取在顺子序列中的索引
    indices = []
    for c in cards:
        idx = _get_straight_index(c.rank)
        if idx == -1:
            return None
        indices.append(idx)

    indices.sort()

    # 检查是否连续且无重复
    if len(set(indices)) != n:
        return None
    if indices[-1] - indices[0] != n - 1:
        return None

    # key_value 使用最小牌的索引作为比较依据
    start_rank = STRAIGHT_RANKS[indices[0]]
    key_value = BASE_RANK_ORDER[start_rank]

    return HandType(HandCategory.STRAIGHT, cards, key_value, n)


def _try_double_straight(cards: list, level_rank: str) -> HandType:
    """尝试识别双龙（连对）"""
    n = len(cards)
    if n < 6 or n % 2 != 0:
        return None

    # 不能包含2、王、级牌
    for c in cards:
        if c.rank == '2' or c.is_joker() or c.rank == level_rank:
            return None

    # 统计每个点数的牌数
    rank_count = {}
    for c in cards:
        rank_count[c.rank] = rank_count.get(c.rank, 0) + 1

    # 每个点数必须恰好2张
    for count in rank_count.values():
        if count != 2:
            return None

    pair_count = len(rank_count)
    if pair_count < 3:
        return None

    # 检查点数在顺子序列中是否连续
    indices = []
    for rank in rank_count:
        idx = _get_straight_index(rank)
        if idx == -1:
            return None
        indices.append(idx)

    indices.sort()
    if indices[-1] - indices[0] != pair_count - 1:
        return None

    start_rank = STRAIGHT_RANKS[indices[0]]
    key_value = BASE_RANK_ORDER[start_rank]

    return HandType(HandCategory.DOUBLE_STRAIGHT, cards, key_value, pair_count)


def can_beat(played: HandType, current: HandType) -> bool:
    """
    判断 current 是否能管住 played
    played: 场上已出的牌型
    current: 当前想出的牌型
    返回: True 表示能管住
    """
    if played is None:
        return True

    pc = played.category
    cc = current.category

    # 四幺四管一切
    if cc == HandCategory.SI_YAO_SI:
        return True

    # 被四幺四管的牌无法再被其他牌管
    if pc == HandCategory.SI_YAO_SI:
        return False

    # === 双龙特殊规则（必须在双王之前判断）===
    # 双龙只能被四幺四管，其他任何牌型（包括双王）都不能管双龙
    if pc == HandCategory.DOUBLE_STRAIGHT:
        if cc == HandCategory.DOUBLE_STRAIGHT:
            return current.length == played.length and current.key_value > played.key_value
        return False

    # 双王管除四幺四和双龙外的一切
    if cc == HandCategory.JOKER_BOMB:
        return True
    if pc == HandCategory.JOKER_BOMB:
        return False

    # 双龙可以管任意单龙（不要求对数等于单龙张数）
    if cc == HandCategory.DOUBLE_STRAIGHT:
        if pc == HandCategory.STRAIGHT:
            return True
        return False

    # === 炸弹规则（炸/轰可以管非炸弹非双龙牌型）===
    is_played_bomb = pc in (HandCategory.BOMB3, HandCategory.BOMB4)
    is_current_bomb = cc in (HandCategory.BOMB3, HandCategory.BOMB4)

    # 轰 > 炸
    if cc == HandCategory.BOMB4:
        if pc == HandCategory.BOMB4:
            return current.key_value > played.key_value
        if pc == HandCategory.BOMB3:
            return True
        # 轰管非炸弹非双龙
        return True

    if cc == HandCategory.BOMB3:
        if pc == HandCategory.BOMB3:
            return current.key_value > played.key_value
        if pc == HandCategory.BOMB4:
            return False
        # 炸管非炸弹非双龙
        return True

    # 如果场上是炸弹，普通牌型无法管
    if is_played_bomb:
        return False

    # === 同类型比较 ===
    if pc != cc:
        return False

    if pc == HandCategory.SINGLE:
        return current.key_value > played.key_value

    if pc == HandCategory.PAIR:
        return current.key_value > played.key_value

    if pc == HandCategory.STRAIGHT:
        return current.length == played.length and current.key_value > played.key_value

    return False


def is_bomb_type(category: HandCategory) -> bool:
    """判断是否为炸弹类型（炸、轰、双王、四幺四）"""
    return category in (
        HandCategory.BOMB3, HandCategory.BOMB4,
        HandCategory.JOKER_BOMB, HandCategory.SI_YAO_SI,
    )
