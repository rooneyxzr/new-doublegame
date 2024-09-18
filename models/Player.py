import HandInfo

class Player:
    def __init__(self, name, hand_info, model):
        self.name = name
        self.hand_info = hand_info
        self.model = model

    def genNextMove(self):
        