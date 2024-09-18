
mappingIndex = {'1': 0, '2':1, '3':2, '4':3, '5':4,
           '6':5, '7':6, '8':7, '9':8, 'z':9, 'f':10, 'b':11}

mappingName = {0:'1', 1:'2', 2:'3', 3:'4', 4:'5', 5:'6',
               6:'7', 7:'8', 8:'9', 9:'z', 10:'f', 11:'b'}

def getName(name):
    return mappingName[name]

def getIndex(index):
    return mappingIndex[index]

def is_hand_finished(hand):
    pair_index = -1
    rt = 0
    for i in range(9):
        rt += hand[i] * (i+1)
    for i in range(9,12):
        rt += hand[i] * 10
    # print("rewardcal: " + hand.__str__())
    if (hand[9] + hand[10] + hand[11] == 0):
        rt += 10
    if (hand[9] + hand[10] + hand[11] == 5):
        rt += 40
    
    # Check for pairs that could form a complete hand
    for index in range(len(hand)):
        if hand[index] >= 2:
            if pair_index != -1:
                if (hand[pair_index] == 2 and hand[index] == 3) or (hand[pair_index] == 3 and hand[index] == 2):
                    return rt + 20
                else:
                    return -8
            else:
                pair_index = index
    
    # If no pair found, the hand is not finished
    if pair_index == -1:
        return -1
    
    # Check for sequences that can be formed with the pair
    hand[pair_index] -= 2
    for i in range(len(hand) - 2):
        if hand[i] == 1 and hand[i + 1] == 1 and hand[i + 2] == 1:
            hand[pair_index] += 2
            return rt
    
    # If no sequence found, the hand is not finished
    hand[pair_index] += 2
    return -1

def showHand(hand):
    s = ''
    for i in range(12):
        for j in range(hand[i]):
            s += getName(i)
    return s