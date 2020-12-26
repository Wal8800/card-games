import numpy as np
from playingcards.card import Card, Suit, Rank
from bigtwo import BigTwo
from gym import spaces
import random
from sklearn.preprocessing import OneHotEncoder
from collections import Counter

suit_encoder = OneHotEncoder(sparse=False, categories='auto')
temp = np.array(range(4)).reshape(4, 1)
suit_one_hot = suit_encoder.fit_transform(temp)
rank_encoder = OneHotEncoder(sparse=False, categories='auto')
rank_one_hot = rank_encoder.fit_transform(np.array(range(13)).reshape(13, 1))


def pick_cards_from_hand(raw_action, hand):
    action = []
    for idx, value in enumerate(raw_action):
        if value == 1:
            action.append(hand[idx])

    return action


def to_rank_suit_obs(card_numbers):
    suit_rank_obs = np.array([])
    for card in card_numbers:
        rank = card % 13
        suit = card // 13

        suit_rank_obs = np.concatenate([suit_rank_obs, [suit, rank]])

    return suit_rank_obs


def to_one_hot_encoding(card_numbers):
    one_hot_obs = np.array([])
    for card in card_numbers:
        rank = card % 13
        suit = card // 13

        suit_array = suit_one_hot[suit]
        rank_array = rank_one_hot[rank]
        one_hot_obs = np.concatenate([one_hot_obs, suit_array, rank_array])

    return one_hot_obs


def from_one_hot_encoding(obs):
    suit_arr_list = [obs[x:x + 4] for x in range(0, 221, 17)]
    rank_arr_list = [obs[x:x + 13] for x in range(4, 221, 17)]
    suit_rank_list = list(
        zip(
            suit_encoder.inverse_transform(suit_arr_list).flatten(),
            rank_encoder.inverse_transform(rank_arr_list).flatten()
        )
    )
    hand = []
    for pair in suit_rank_list:
        hand.append((pair[0] * 13) + pair[1])

    return [Card.from_number(x) for x in hand]


def random_hand():
    return random.sample(range(52), 13)


class ValidCardGame:
    def __init__(self):
        self.current_hand = None
        self.observation_space = spaces.Box(low=0, high=51, shape=(13,), dtype=np.int32)
        self.action_space = spaces.MultiBinary(13)
        self.past_result = []

    def step(self, action):
        hand = [Card.from_number(x) for x in self.current_hand]
        cards = pick_cards_from_hand(action, hand)
        valid_card = False
        if len(cards) > 0:
            valid_card = BigTwo.is_valid_card_combination(cards)

        reward = 0
        if len(cards) == 1 or len(cards) == 2 or len(cards) == 5:
            reward += len(cards) * len(cards)

        if valid_card and len(cards) > 1:
            reward += len(cards) * len(cards)

        self.past_result.append((self.current_hand, cards, reward))

        self.current_hand = random_hand()
        return self.current_hand, reward, reward == 0, {}

    def reset(self):
        self.current_hand = random_hand()
        self.past_result = []

        return self.current_hand


class ValidCardGameAlt:
    def __init__(self):
        obs_shape = self.reset()
        # 13 cards * 4 suit + 13 * 13 type of cards + 13 + 4 (Count of each rank and suit)
        self.observation_space = spaces.Box(low=min(obs_shape), high=12, shape=(len(obs_shape),), dtype=np.int32)
        self.action_space = spaces.MultiBinary(13)
        self.reward_range = (0, 25)
        self.metadata = None
        self.past_result = []
        self.spec = None

    def step(self, action):
        hand = [Card.from_number(x) for x in self.current_hand]
        cards = pick_cards_from_hand(action, hand)
        valid_card = False
        if len(cards) > 0:
            valid_card = BigTwo.is_valid_card_combination(cards)

        reward = len(cards) * len(cards) if valid_card and len(cards) > 1 else 0

        self.past_result.append((self.current_hand, cards, reward))
        return self._create_mapping_obs(), reward, reward <= 0, {}

    def reset(self):
        self.past_result = []

        return self._create_mapping_obs()

    def _create_obs(self):
        self.current_hand = random_hand()
        self.current_hand.sort()
        obs = to_one_hot_encoding(self.current_hand)

        rank_counts = Counter([Card.from_number(card_number).rank for card_number in self.current_hand])
        suit_counts = Counter([Card.from_number(card_number).suit for card_number in self.current_hand])

        total_rank_counts = [rank_counts[rank] if rank in rank_counts else 0 for rank in Rank]
        total_suit_counts = [suit_counts[suit] if suit in suit_counts else 0 for suit in Suit]

        return np.concatenate([obs, total_rank_counts, total_suit_counts])

    def _create_mapping_obs(self):
        self.current_hand = random_hand()
        self.current_hand.sort()

        rank_mapping = [[1 if Card.from_number(card_number).rank == rank else 0 for card_number in self.current_hand]
                        for rank in Rank]
        suit_mapping = [[1 if Card.from_number(card_number).suit == suit else 0 for card_number in self.current_hand]
                        for suit in Suit]

        return np.concatenate([rank_mapping, suit_mapping]).flatten()

    def current_hand_card(self):
        return [Card.from_number(card_number) for card_number in self.current_hand]


if __name__ == '__main__':
    game = ValidCardGameAlt()
    obs = game.reset()
    print(obs)
