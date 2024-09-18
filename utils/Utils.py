# 这个字典将字符串映射到索引，用于表示麻将牌
mappingIndex = {str(i): i-1 for i in range(1, 10)}
mappingIndex.update({'z': 9, 'f': 10, 'b': 11})

# 这个字典是mappingIndex的反向映射，用于从索引获取牌面
mappingName = {v: k for k, v in mappingIndex.items()}

def getName(name):
    # 根据索引获取牌面
    return mappingName[name]

def getIndex(index):
    # 根据牌面获取索引
    return mappingIndex[index]

def is_hand_finished(hand):
    # 检查手牌数量是否为5张
    if sum(hand) != 5:
        return -1, "手牌数量不正确"

    # 计算基本分
    base_score = sum(i * hand[i-1] for i in range(1, 10)) + sum(10 * hand[i] for i in range(9, 12))

    # 检查特殊牌型
    is_all_pongs = all(count in [0, 3] for count in hand)  # 碰碰胡
    is_all_honors = sum(hand[:9]) == 0  # 字一色
    is_all_numbers = sum(hand[9:]) == 0  # 清一色

    # 寻找对子、刻子和顺子
    pairs = []
    pongs = []
    hand_copy = hand.copy()

    # 检查对子和刻子
    for i, count in enumerate(hand_copy):
        if count == 2:
            pairs.append(i)
        elif count == 3:
            pongs.append(i)
            hand_copy[i] = 0

    # 检查顺子（仅适用于数字牌）
    for i in range(7):
        while hand_copy[i] > 0 and hand_copy[i+1] > 0 and hand_copy[i+2] > 0:
            hand_copy[i] -= 1
            hand_copy[i+1] -= 1
            hand_copy[i+2] -= 1

    # 判断是否胡牌
    if sum(hand_copy) == 0 and len(pairs) == 1:
        # 计算额外分数
        extra_score = 0
        if is_all_pongs:
            extra_score += 20  # 碰碰胡
        if is_all_honors:
            extra_score += 50  # 字一色
        elif is_all_numbers:
            extra_score += 10  # 清一色

        total_score = base_score + extra_score
        return total_score, "胡牌"
    else:
        return -1, "未胡牌"

def showHand(hand):
    # 将手牌转换为字符串表示
    return ''.join(getName(i) * count for i, count in enumerate(hand))