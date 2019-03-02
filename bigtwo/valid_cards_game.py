import numpy as np
from playingcards.card import Card
from bigtwo import BigTwo
from gym import spaces


def pick_cards_from_hand(raw_action, hand):
    action = []
    for idx, value in enumerate(raw_action):
        if value == 1:
            action.append(hand[idx])

    return action


class ValidCardGame:
    def __init__(self):
        self.current_hand = None
        self.observation_space = spaces.Box(low=-1, high=51, shape=(13,), dtype=np.int32)
        self.action_space = spaces.MultiBinary(13)
        self.past_result = []

    def step(self, action):
        hand = [Card.from_number(x) for x in self.current_hand]
        cards = pick_cards_from_hand(action, hand)
        valid_card = False
        if len(cards) > 0:
            valid_card = BigTwo.is_valid_card_combination(cards)
        reward = len(cards)*len(cards) if valid_card else 0

        self.past_result.append((self.current_hand, cards, reward))

        self.current_hand = self.random_hand()
        return self.random_hand(), reward, reward == 0, {}

    def reset(self):
        self.current_hand = ValidCardGame.random_hand()
        self.past_result = []
        return self.current_hand

    @staticmethod
    def random_hand():
        return np.random.randint(52, size=13)
