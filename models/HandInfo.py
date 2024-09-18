import numpy as np
import utils.Utils as Utils

class HandInfo:
    def __init__(self):
        # 初始化一个长度为12的整数数组来表示手牌
        self.hand = np.zeros(12, dtype=int)
        # 初始化一个长度为12的整数数组来表示剩余的牌
        self.left = np.zeros(12, dtype=int)
        # 一系列布尔值来表示不同的游戏状态
        self.isConfirmed = False  # 是否确认
        self.isConfirmedAtStart = False  # 是否在开始时确认
        self.isTianho = False  # 是否天和
        self.isDiho = False  # 是否地和
        self.isWinBySelf = False  # 是否自摸
        self.isPeng = False  # 是否碰牌

    def getNext(self, newPiece):
        # 增加一张新牌到手牌中
        self.hand[newPiece] += 1

    def outNext(self, oldPiece):
        # 从手牌中移除一张牌
        self.hand[oldPiece] -= 1

    def isTingableFour(self):
        # 检查是否听牌（四张牌的情况）
        for i in range(12):
            self.hand[i] += 1
            if Utils.is_hand_finished(self.hand):
                self.hand[i] -= 1
                return True
            self.hand[i] -= 1
        return False

    def isTingableFive(self):
        # 检查是否听牌（五张牌的情况）
        for i in range(12):
            if self.hand[i] > 0:
                self.hand[i] -= 1
                if self.isTingableFour():
                    self.hand[i] += 1
                    return True
                self.hand[i] += 1
        return False

    def isTinghoed(self):
        # 检查是否已经和牌
        return Utils.is_hand_finished(self.hand)

    def showHand(self):
        # 打印当前手牌
        print(''.join(Utils.getName(i) * count for i, count in enumerate(self.hand)))