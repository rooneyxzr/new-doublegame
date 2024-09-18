import numpy as np
import utils.Utils as Utils

class HandInfo:
    def __init__(self):
        self.hand = np.zeros(12, dtype = int)
        self.left = np.zeros(12, dtype = int)
        self.isConfirmed = False
        self.isConfirmedAtStart = False
        self.isTianho = False
        self.isDiho = False
        self.isWinBySelf = False
        self.isPeng = False
        for i in range(12):
            self.hand[i] = 0

    def getNext(self, newPiece):
        self.hand[newPiece]+=1

    def outNext(self, oldPiece):
        self.hand[oldPiece]-=1

    def isTingableFour(self):
        for i in range(12):
            self.hand[i] += 1
            if (Utils.is_hand_finished(self)):
                self.hand[i] -= 1
                return True
            else:
                self.hand[i] -= 1
        return False

    def isTingableFive(self):
        for i in range(12):
            if (self.hand[i] > 0):
                self.hand[i] -= 1
                if (self.isTingableFour() == True):
                    return True
                self.hand[i] += 1
        return False

    def isTinghoed(self):
        if (Utils.is_hand_finished(self)):
            return True
        return False

    def showHand(self):
        s = ''
        for i in range(12):
            for j in range(self.hand[i]):
                s += Utils.getName(i)
        print(s)