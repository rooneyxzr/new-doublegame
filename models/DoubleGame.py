import random
import numpy as np
import HandInfo
import utils.Utils as Utils


class DoubleGameBoard:
    def __init__(self, value):
        self.totalPieces = np.zeros(12)
        self.value = value

    def get_state(self):
        return self.playerhandinfo.hand.flatten()
    
handInfo = HandInfo.HandInfo()
for i in range(5):
    handInfo.getNext(random.randint(0,11))
handInfo.showHand()
print(Utils.is_hand_finished(handInfo))