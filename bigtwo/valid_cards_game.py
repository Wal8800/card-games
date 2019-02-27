import numpy as np
from playingcards.card import Card
from bigtwo import BigTwo


def pick_cards_from_hand(raw_action, hand):
    action = []
    for idx, value in enumerate(raw_action):
        if value == 1:
            action.append(hand[idx])

    return action


class ValidCardGame:
    def __init__(self):
        self.current_hand = None

    def step(self, action):
        hand = [Card.from_number(x) for x in self.current_hand]
        cards = pick_cards_from_hand(action, hand)
        valid_card = False
        if len(cards) > 0:
            valid_card = BigTwo.is_valid_card_combination(cards)
        reward = 1 if valid_card else 0
        return self.random_hand(), reward, reward == 0

    def reset(self):
        self.current_hand = ValidCardGame.random_hand()
        return self.current_hand

    @staticmethod
    def random_hand():
        return np.random.randint(52, size=13)