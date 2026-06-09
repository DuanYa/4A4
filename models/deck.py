"""
牌组管理：洗牌与发牌
"""
import random
from models.card import create_full_deck, sort_cards


def shuffle_and_deal(level_rank: str) -> list:
    """
    洗牌并发牌给4个玩家
    54张牌分配：玩家0和玩家2各14张，玩家1和玩家3各13张
    返回: [hand0, hand1, hand2, hand3]，每个hand是排好序的Card列表
    """
    deck = create_full_deck()
    random.shuffle(deck)

    # 分配: 14, 13, 14, 13
    hands = [[], [], [], []]
    distribution = [14, 13, 14, 13]

    idx = 0
    for player_idx in range(4):
        count = distribution[player_idx]
        hands[player_idx] = deck[idx:idx + count]
        idx += count

    # 对每个玩家的手牌排序
    for i in range(4):
        hands[i] = sort_cards(hands[i], level_rank)

    return hands
