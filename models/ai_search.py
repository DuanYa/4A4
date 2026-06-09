"""
AI出牌搜索逻辑 - 从手牌中搜索合法的出牌组合
通过mixin方式注入到AIPlayer类
"""
from models.card import STRAIGHT_RANKS
from models.hand_type import (
    identify_hand, can_beat, HandType, HandCategory
)


def _find_free_play(self):
    """自由出牌：有单龙优先出单龙，否则在可清空时出双龙，最后出最小单张"""
    if not self.hand:
        return None

    # 首家出牌时双龙有限制，因此优先寻找合法单龙，方便验证龙牌型
    max_len = min(len(self.hand), len(STRAIGHT_RANKS))
    for length in range(max_len, 2, -1):
        straights = _get_straights(self, length)
        if straights:
            return straights[0]

    # 双龙首出只有清空手牌时合法
    if len(self.hand) >= 6 and len(self.hand) % 2 == 0:
        pair_count = len(self.hand) // 2
        dbls = _get_dbl_straights(self, pair_count)
        for indices in dbls:
            if len(indices) == len(self.hand):
                return indices

    return [len(self.hand) - 1]


def _find_beat_play(self, last_ht):
    """找能管住上家的牌"""
    if not self.hand:
        return None
    cat = last_ht.category

    # 上家出单龙时，优先尝试用同长度双龙管，便于验证双龙规则
    if cat == HandCategory.STRAIGHT:
        dbl = _find_double_straight_for_straight(self, last_ht)
        if dbl is not None:
            return dbl

    # 同类型跟牌
    r = _find_same_type(self, cat, last_ht)
    if r is not None:
        return r

    # 非炸弹、非双龙场合，尝试用炸弹管
    no_bomb_cats = (
        HandCategory.SINGLE, HandCategory.PAIR,
        HandCategory.STRAIGHT)
    if cat in no_bomb_cats:
        b = _find_bomb(self, last_ht)
        if b is not None:
            return b

    # 炸弹之间比较
    if cat in (HandCategory.BOMB3, HandCategory.BOMB4):
        return _find_bomb(self, last_ht)

    return None


def _find_double_straight_for_straight(self, last_ht):
    """上家出单龙时，优先寻找任意合法双龙来管"""
    hand = self.hand
    lr = self.level_rank
    max_pairs = len(self.hand) // 2
    for pair_count in range(3, max_pairs + 1):
        for indices in _get_dbl_straights(self, pair_count):
            cards = [hand[i] for i in indices]
            ht = identify_hand(cards, lr)
            if ht and can_beat(last_ht, ht):
                return indices
    return None


def _find_same_type(self, cat, last_ht):
    """找同类型中能管住的最小组合"""
    hand = self.hand
    lr = self.level_rank

    if cat == HandCategory.SINGLE:
        for i in range(len(hand) - 1, -1, -1):
            ht = identify_hand([hand[i]], lr)
            if ht and can_beat(last_ht, ht):
                return [i]

    elif cat == HandCategory.PAIR:
        for indices in _get_pairs(self):
            cards = [hand[i] for i in indices]
            ht = identify_hand(cards, lr)
            if ht and can_beat(last_ht, ht):
                return indices

    elif cat == HandCategory.STRAIGHT:
        for indices in _get_straights(self, last_ht.length):
            cards = [hand[i] for i in indices]
            ht = identify_hand(cards, lr)
            if ht and can_beat(last_ht, ht):
                return indices

    elif cat == HandCategory.DOUBLE_STRAIGHT:
        for indices in _get_dbl_straights(self, last_ht.length):
            cards = [hand[i] for i in indices]
            ht = identify_hand(cards, lr)
            if ht and can_beat(last_ht, ht):
                return indices

    return None


def _find_bomb(self, last_ht):
    """找能管住的炸弹"""
    hand = self.hand
    lr = self.level_rank

    # 炸（三条）
    for indices in _get_n_kind(self, 3):
        cards = [hand[i] for i in indices]
        ht = identify_hand(cards, lr)
        if ht and can_beat(last_ht, ht):
            return indices

    # 轰（四条）
    for indices in _get_n_kind(self, 4):
        cards = [hand[i] for i in indices]
        ht = identify_hand(cards, lr)
        if ht and can_beat(last_ht, ht):
            return indices

    # 双王
    jk = _get_joker_bomb(self)
    if jk:
        cards = [hand[i] for i in jk]
        ht = identify_hand(cards, lr)
        if ht and can_beat(last_ht, ht):
            return jk

    # 四幺四
    sisi = _get_si_yao_si(self)
    if sisi:
        return sisi

    return None


# ==================== 手牌分析 ====================

def _group_by_rank(self):
    groups = {}
    for i, c in enumerate(self.hand):
        groups.setdefault(c.rank, []).append(i)
    return groups


def _get_pairs(self):
    """所有对子索引，按牌力从小到大"""
    groups = _group_by_rank(self)
    pairs = []
    for rank, idx in groups.items():
        if len(idx) >= 2 and rank not in ('BJ', 'RJ'):
            pairs.append(idx[:2])
    pairs.sort(key=lambda x:
               self.hand[x[0]].get_value(self.level_rank))
    return pairs


def _get_n_kind(self, n):
    """所有n张同点数，按牌力从小到大"""
    groups = _group_by_rank(self)
    result = []
    for rank, idx in groups.items():
        if len(idx) >= n and rank not in ('BJ', 'RJ'):
            result.append(idx[:n])
    result.sort(key=lambda x:
                self.hand[x[0]].get_value(self.level_rank))
    return result


def _get_joker_bomb(self):
    bj = rj = -1
    for i, c in enumerate(self.hand):
        if c.rank == 'BJ':
            bj = i
        elif c.rank == 'RJ':
            rj = i
    if bj >= 0 and rj >= 0:
        return [bj, rj]
    return None


def _get_si_yao_si(self):
    fours = [i for i, c in enumerate(self.hand)
             if c.rank == '4' and not c.is_joker()]
    aces = [i for i, c in enumerate(self.hand)
            if c.rank == 'A' and not c.is_joker()]
    if len(fours) >= 2 and len(aces) >= 1:
        return fours[:2] + aces[:1]
    return None


def _get_straights(self, length):
    """找所有指定长度的单龙"""
    results = []
    rk2idx = {}
    for i, c in enumerate(self.hand):
        if (c.rank == '2' or c.is_joker()
                or c.rank == self.level_rank):
            continue
        if c.rank in STRAIGHT_RANKS:
            pos = STRAIGHT_RANKS.index(c.rank)
            if pos not in rk2idx:
                rk2idx[pos] = []
            rk2idx[pos].append(i)

    positions = sorted(rk2idx.keys())
    for si in range(len(positions)):
        seq = []
        exp = positions[si]
        for j in range(si, len(positions)):
            if positions[j] == exp:
                seq.append(positions[j])
                exp += 1
            else:
                break
            if len(seq) == length:
                indices = [rk2idx[p][0] for p in seq]
                results.append(indices)
                break
    results.sort(key=lambda x:
                 self.hand[x[0]].get_value(self.level_rank))
    return results


def _get_dbl_straights(self, pair_count):
    """找所有指定对数的双龙"""
    results = []
    rk2idx = {}
    for i, c in enumerate(self.hand):
        if (c.rank == '2' or c.is_joker()
                or c.rank == self.level_rank):
            continue
        if c.rank in STRAIGHT_RANKS:
            pos = STRAIGHT_RANKS.index(c.rank)
            if pos not in rk2idx:
                rk2idx[pos] = []
            rk2idx[pos].append(i)

    valid = sorted([p for p, idx in rk2idx.items()
                    if len(idx) >= 2])
    for si in range(len(valid)):
        seq = []
        exp = valid[si]
        for j in range(si, len(valid)):
            if valid[j] == exp:
                seq.append(valid[j])
                exp += 1
            else:
                break
            if len(seq) == pair_count:
                indices = []
                for p in seq:
                    indices.extend(rk2idx[p][:2])
                results.append(indices)
                break
    results.sort(key=lambda x:
                 self.hand[x[0]].get_value(self.level_rank))
    return results


def _reconstruct_ht(ht_dict):
    """从字典重建HandType"""
    from models.ai_player import card_from_dict
    cards = [card_from_dict(c) for c in ht_dict['cards']]
    cat = HandCategory[ht_dict['category']]
    return HandType(cat, cards,
                    ht_dict.get('key_value', 0),
                    ht_dict.get('length', 0))
