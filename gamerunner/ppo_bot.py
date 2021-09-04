import itertools
from typing import Dict, List, Tuple, Mapping, Optional, Union, Iterable

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers

from algorithm.agent import (
    PPOAgent,
    PPOBufferInterface,
    discounted_sum_of_rewards,
    get_mlp_vf,
    SavedPPOAgent,
)
from bigtwo.bigtwo import BigTwoObservation, BigTwo, BigTwoHand
from gamerunner.big_two_bot import BigTwoBot
from playingcards.card import Card, Suit, Rank

suit_ohe = {
    Suit.spades: [0, 0, 0, 1],
    Suit.hearts: [0, 0, 1, 0],
    Suit.clubs: [0, 1, 0, 0],
    Suit.diamond: [1, 0, 0, 0],
}

rank_ohe = {
    Rank.ace: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    Rank.two: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    Rank.three: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    Rank.four: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    Rank.five: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    Rank.six: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    Rank.seven: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    Rank.eight: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    Rank.nine: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    Rank.ten: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    Rank.jack: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    Rank.queen: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    Rank.king: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

empty_suit = [0, 0, 0, 0]
empty_rank = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

PAST_CARDS_PLAYED = "past_cards_played"
CURRENT_OBS = "current_obs"


def obs_to_arr(obs: BigTwoObservation) -> np.ndarray:
    data = []
    data += obs.num_card_per_player

    for i in range(5):
        # add place holder for empty card
        if i >= len(obs.last_cards_played):
            data += [-1, -1]
            continue

        curr_card = obs.last_cards_played[i]
        data += [Card.SUIT_NUMBER[curr_card.suit], Card.RANK_NUMBER[curr_card.rank]]

    for i in range(13):
        if i >= len(obs.your_hands):
            data += [-1, -1]
            continue

        curr_card = obs.your_hands[i]
        data += [Card.SUIT_NUMBER[curr_card.suit], Card.RANK_NUMBER[curr_card.rank]]

    return np.array(data)


def cards_to_ohe(cards: List[Card], padded_length: int) -> List[int]:
    if len(cards) > padded_length:
        raise ValueError("expected card length to be less or equal to padded length")
    data = []

    for i in range(padded_length):
        # add place holder for empty card
        if i >= len(cards):
            data += empty_suit
            data += empty_rank
            continue

        curr_card = cards[i]
        data += suit_ohe[curr_card.suit]
        data += rank_ohe[curr_card.rank]

    return data


def obs_to_ohe(obs: BigTwoObservation, include_last_cards_played=True) -> np.ndarray:
    data = []
    data += obs.num_card_per_player

    # first turn
    data.append(1 if len(obs.last_cards_played) == 0 else 0)

    # current player number
    current_player_number = [0, 0, 0, 0]
    current_player_number[obs.current_player] = 1

    data += current_player_number

    # last player number
    last_player_number = [0, 0, 0, 0]
    if obs.last_player_played <= 4:
        last_player_number[obs.last_player_played] = 1
    data += last_player_number

    # cards length
    data.append(len(obs.last_cards_played))

    if include_last_cards_played:
        data += cards_to_ohe(obs.last_cards_played, 5)
    data += cards_to_ohe(obs.your_hands.cards, 13)

    return np.array(data)


def create_action_cat_mapping(
    num_cards_in_hand=13,
) -> Tuple[Dict[int, List[int]], Dict[frozenset, int]]:
    """
    :param num_cards_in_hand:
    :return: a map of category to raw action (one hot encoded) and also a map for reverese lookup of indices to category
    """
    result = {}
    cat = 0
    reverse_lookup = {}

    # skip action
    result[cat] = [0] * num_cards_in_hand
    cat += 1

    # play single card action
    for i in range(num_cards_in_hand):
        temp = [0] * num_cards_in_hand
        temp[i] = 1
        result[cat] = temp

        reverse_lookup[frozenset([i])] = cat
        cat += 1

    # play double card action
    for pair_idx in itertools.combinations(range(num_cards_in_hand), 2):
        temp = [0] * num_cards_in_hand
        temp[pair_idx[0]] = 1
        temp[pair_idx[1]] = 1
        reverse_lookup[frozenset(pair_idx)] = cat
        result[cat] = temp
        cat += 1

    # play combinations action
    for comb_idx in itertools.combinations(range(num_cards_in_hand), 5):
        temp = [0] * num_cards_in_hand
        reverse_lookup[frozenset(comb_idx)] = cat
        for idx in comb_idx:
            temp[idx] = 1
        result[cat] = temp
        cat += 1

    return result, reverse_lookup


def generate_action_mask_first_turn(
    idx_cat_mapping,
    obs: BigTwoObservation,
) -> np.array:
    result = np.full(len(idx_cat_mapping) + 1, False)

    diamond_three = Card(Suit.diamond, Rank.three)

    card_idx_mapping = {}
    for idx in range(len(obs.your_hands)):
        card = obs.your_hands[idx]
        card_idx_mapping[card] = idx

        if card != diamond_three:
            continue

        cat = idx_cat_mapping[frozenset([idx])]
        result[cat] = True

    for pair in obs.your_hands.pairs:
        if diamond_three not in pair:
            continue

        pair_idx = [card_idx_mapping.get(card) for card in pair]
        cat = idx_cat_mapping[frozenset(pair_idx)]
        result[cat] = True

    for _, combinations in obs.your_hands.combinations.items():
        for comb in combinations:
            if diamond_three not in comb:
                continue

            comb_idx = [card_idx_mapping.get(card) for card in comb]
            cat = idx_cat_mapping[frozenset(comb_idx)]
            result[cat] = True

    return result


# if play anything update all the mapping
def generate_all_option(obs: BigTwoObservation):
    card_idx_mapping = {obs.your_hands[idx]: idx for idx in range(len(obs.your_hands))}
    single_options = [[idx] for idx in range(len(obs.your_hands))]
    pair_options = [
        [card_idx_mapping.get(card) for card in pair] for pair in obs.your_hands.pairs
    ]
    combination_options = []
    for _, combinations in obs.your_hands.combinations.items():
        for comb in combinations:
            comb_idx = [card_idx_mapping.get(card) for card in comb]
            combination_options.append(comb_idx)

    return single_options + pair_options + combination_options


def generate_single_options(hands: BigTwoHand, target_to_beat: Card) -> List[List[int]]:
    """
    :param hands:
    :param target_to_beat:
    :return: a list of indexes, each list of indexes represent the cards to play
    """
    return [
        [idx]
        for idx, card in enumerate(hands)
        if BigTwo.is_bigger([card], [target_to_beat])
    ]


def generate_pair_options(
    hands: BigTwoHand, target_to_beat: List[Card]
) -> List[List[int]]:
    """
    :param hands:
    :param target_to_beat:
    :return: a list of indexes, each list of indexes represent the cards to play
    """
    if len(target_to_beat) != 2:
        raise ValueError("expects target_to_beat to have 2 cards")

    card_idx_mapping = {hands[idx]: idx for idx in range(len(hands))}
    return [
        [card_idx_mapping.get(card) for card in pair]
        for pair in hands.pairs
        if BigTwo.is_bigger(list(pair), target_to_beat)
    ]


def generate_combinations_options(
    hands: BigTwoHand, target_to_beat: List[Card]
) -> List[List[int]]:
    """
    :param hands:
    :param target_to_beat:
    :return: a list of indexes, each list of indexes represent the cards to play
    """
    if len(target_to_beat) != 5:
        raise ValueError("expects target_to_beat to have 5 cards")

    card_idx_mapping = {hands[idx]: idx for idx in range(len(hands))}
    combination_options = []
    for _, combinations in hands.combinations.items():
        for comb in combinations:
            if not BigTwo.is_bigger(comb, target_to_beat):
                continue

            comb_idx = [card_idx_mapping.get(card) for card in comb]
            combination_options.append(comb_idx)
    return combination_options


def generate_action_mask(
    idx_cat_mapping,
    obs: BigTwoObservation,
) -> np.array:
    if obs.is_first_turn():
        return generate_action_mask_first_turn(idx_cat_mapping, obs)

    if obs.can_play_any_cards():
        options = generate_all_option(obs)
    elif len(obs.last_cards_played) == 1:
        options = generate_single_options(obs.your_hands, obs.last_cards_played[0])
    elif len(obs.last_cards_played) == 2:
        options = generate_pair_options(obs.your_hands, obs.last_cards_played)
    elif len(obs.last_cards_played) == 5:
        options = generate_combinations_options(obs.your_hands, obs.last_cards_played)
    else:
        raise ValueError("unexpected scenario to generate invalid actions mask")

    # idx_cat_mapping + 1 beacause we need to include no action
    result = np.full(len(idx_cat_mapping) + 1, False)
    result[0] = obs.can_skip()

    for option in options:
        cat = idx_cat_mapping[frozenset(option)]
        result[cat] = True

    return result


class PlayerBuffer:
    def __init__(self, gamma=0.99, lam=0.95):
        self.obs_buf = []
        self.act_buf = []
        self.adv_buf = []
        self.rew_buf = []
        self.ret_buf = []
        self.logp_buf = []
        self.mask_buf = []
        self.gamma, self.lam = gamma, lam

    def store(self, obs, act, rew, logp, mask):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.logp_buf.append(logp)
        self.mask_buf.append(mask)

    def get_curr_path(self):
        return np.array(self.obs_buf)

    def is_empty(self):
        return len(self.obs_buf) == 0

    def finish_path(self, estimated_vals: np.array, last_val=0.0):
        rews = np.append(self.rew_buf, last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        # δVt = rt + γV(st+1) − V(st)
        vals = np.append(estimated_vals, last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf = discounted_sum_of_rewards(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf = discounted_sum_of_rewards(rews, self.gamma)[:-1]


class MultiInputPlayerBuffer:
    def __init__(self, gamma=0.99, lam=0.95):
        self.obs_buf = {}
        self.act_buf = []
        self.adv_buf = []
        self.rew_buf = []
        self.ret_buf = []
        self.logp_buf = []
        self.mask_buf = []
        self.gamma, self.lam = gamma, lam

    def store(self, obs, act, rew, logp, mask):
        assert isinstance(obs, Mapping)
        if self.is_empty():
            self.obs_buf = obs
        else:
            self.obs_buf = {
                k: np.append(v, obs[k], axis=0) for k, v in self.obs_buf.items()
            }

        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.logp_buf.append(logp)
        self.mask_buf.append(mask)

    def get_curr_path(self):
        return self.obs_buf

    def is_empty(self):
        return len(self.obs_buf) == 0

    def finish_path(self, estimated_vals: np.array, last_val=0.0):
        rews = np.append(self.rew_buf, last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        # δVt = rt + γV(st+1) − V(st)
        vals = np.append(estimated_vals, last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf = discounted_sum_of_rewards(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf = discounted_sum_of_rewards(rews, self.gamma)[:-1]


def set_or_add(
    original: Optional[np.ndarray], new: Union[np.ndarray, Iterable]
) -> np.ndarray:
    if original is None:
        return np.array(new)

    return np.append(original, new, axis=0)


class GameBuffer(PPOBufferInterface):
    def __init__(self):
        self.obs_buf = None
        self.act_buf = None
        self.adv_buf = None
        self.rew_buf = None
        self.ret_buf = None
        self.logp_buf = None
        self.mask_buf = None

    def add(self, pbuf: PlayerBuffer):
        self.obs_buf = set_or_add(self.obs_buf, pbuf.obs_buf)
        self.act_buf = set_or_add(self.act_buf, pbuf.act_buf)
        self.adv_buf = set_or_add(self.adv_buf, pbuf.adv_buf)
        self.rew_buf = set_or_add(self.rew_buf, pbuf.rew_buf)
        self.ret_buf = set_or_add(self.ret_buf, pbuf.ret_buf)
        self.logp_buf = set_or_add(self.logp_buf, pbuf.logp_buf)
        self.mask_buf = set_or_add(self.mask_buf, pbuf.mask_buf)

    def get(self):
        return (
            self.obs_buf,
            self.act_buf,
            self.adv_buf.astype("float32"),
            self.ret_buf.astype("float32"),
            self.logp_buf,
        )

    def __add__(self, other):
        result_buf = GameBuffer()
        result_buf.obs_buf = np.append(self.obs_buf, other.obs_buf, axis=0)
        result_buf.act_buf = np.append(self.act_buf, other.act_buf, axis=0)
        result_buf.adv_buf = np.append(self.adv_buf, other.adv_buf, axis=0)
        result_buf.rew_buf = np.append(self.rew_buf, other.rew_buf, axis=0)
        result_buf.ret_buf = np.append(self.ret_buf, other.ret_buf, axis=0)
        result_buf.logp_buf = np.append(self.logp_buf, other.logp_buf, axis=0)
        result_buf.mask_buf = np.append(self.mask_buf, other.mask_buf, axis=0)

        return result_buf


class MultiInputGameBuffer(PPOBufferInterface):
    def __init__(self):
        self.obs_buf: Mapping = {}
        self.act_buf = None
        self.adv_buf = None
        self.rew_buf = None
        self.ret_buf = None
        self.logp_buf = None
        self.mask_buf = None

    def add(self, pbuf: MultiInputPlayerBuffer):
        if len(self.obs_buf) == 0:
            self.obs_buf = pbuf.obs_buf
        else:
            self.obs_buf = {
                k: np.append(v, pbuf.obs_buf[k], axis=0)
                for k, v in self.obs_buf.items()
            }

        self.act_buf = set_or_add(self.act_buf, pbuf.act_buf)
        self.adv_buf = set_or_add(self.adv_buf, pbuf.adv_buf)
        self.rew_buf = set_or_add(self.rew_buf, pbuf.rew_buf)
        self.ret_buf = set_or_add(self.ret_buf, pbuf.ret_buf)
        self.logp_buf = set_or_add(self.logp_buf, pbuf.logp_buf)
        self.mask_buf = set_or_add(self.mask_buf, pbuf.mask_buf)

    def get(self):
        return (
            {k: np.array(v) for k, v in self.obs_buf.items()},
            self.act_buf,
            self.adv_buf.astype("float32"),
            self.ret_buf.astype("float32"),
            self.logp_buf,
        )

    def __add__(self, other):
        result_buf = MultiInputGameBuffer()
        result_buf.obs_buf = {
            k: np.append(v, other.obs_buf[k], axis=0) for k, v in self.obs_buf.items()
        }
        result_buf.act_buf = np.append(self.act_buf, other.act_buf, axis=0)
        result_buf.adv_buf = np.append(self.adv_buf, other.adv_buf, axis=0)
        result_buf.rew_buf = np.append(self.rew_buf, other.rew_buf, axis=0)
        result_buf.ret_buf = np.append(self.ret_buf, other.ret_buf, axis=0)
        result_buf.logp_buf = np.append(self.logp_buf, other.logp_buf, axis=0)
        result_buf.mask_buf = np.append(self.mask_buf, other.mask_buf, axis=0)

        return result_buf


class PPOAction:
    def __init__(
        self, raw_action, action_cat, transformed_obs, action_mask, logp, cards
    ):
        self.transformed_obs = transformed_obs
        self.raw = raw_action
        self.cat = action_cat
        self.mask = action_mask
        self.logp = logp
        self.cards = cards


class RandomPPOBot(BigTwoBot):
    def __init__(self):
        self.action_cat_mapping, self.idx_cat_mapping = create_action_cat_mapping()

    def action(self, observation: BigTwoObservation) -> PPOAction:
        action_mask = generate_action_mask(self.idx_cat_mapping, observation)

        # nonzero return a tuple of 1
        valid_moves = action_mask.nonzero()[0]
        action_cat = np.random.choice(valid_moves)
        raw = self.action_cat_mapping[action_cat]
        cards = [c for idx, c in enumerate(observation.your_hands) if raw[idx] == 1]

        return PPOAction(
            transformed_obs=None,
            raw_action=raw,
            action_cat=action_cat,
            action_mask=action_mask,
            logp=None,
            cards=cards,
        )


class SimplePPOBot(BigTwoBot):
    def __init__(self, observation: BigTwoObservation, lr=0.0001, clip_ratio=0.3):
        self.action_cat_mapping, self.idx_cat_mapping = create_action_cat_mapping()

        self.agent = self._create_agent(observation, lr, clip_ratio)

    def set_weights(self, policy_weights, value_weights):
        self.agent.set_weights(policy_weights, value_weights)

    def get_weights(self):
        return self.agent.get_weights()

    def action(self, observation: BigTwoObservation) -> PPOAction:
        transformed_obs = self.transform_obs(observation)
        action_mask = generate_action_mask(self.idx_cat_mapping, observation)

        action_tensor, logp_tensor = self.agent.action(
            obs=np.array([transformed_obs]), mask=action_mask
        )
        action_cat = action_tensor.numpy()

        raw = self.action_cat_mapping[action_cat]

        cards = [c for idx, c in enumerate(observation.your_hands) if raw[idx] == 1]

        return PPOAction(
            transformed_obs=transformed_obs,
            raw_action=raw,
            action_cat=action_cat,
            action_mask=action_mask,
            logp=logp_tensor.numpy(),
            cards=cards,
        )

    def _create_agent(self, observation: BigTwoObservation, lr=0.0001, clip_ratio=0.3):
        n_action = len(self.action_cat_mapping)
        result = obs_to_ohe(observation)
        obs_inp = layers.Input(shape=result.shape, name="obs")
        x = layers.Dense(512, activation="relu")(obs_inp)
        x = layers.Dense(n_action, activation="relu")(x)
        output = layers.Dense(n_action)(x)
        policy = keras.Model(inputs=obs_inp, outputs=output)

        return PPOAgent(
            policy,
            get_mlp_vf(result.shape, hidden_units=256),
            "BigTwo",
            n_action,
            policy_lr=lr,
            value_lr=lr,
            clip_ratio=clip_ratio,
        )

    def predict_value(self, observations):
        return self.agent.predict_value(observations)

    def update(self, buffer, mini_batch_size: int):
        return self.agent.update(buffer, mini_batch_size, mask_buf=buffer.mask_buf)

    def transform_obs(self, obs: BigTwoObservation):
        return obs_to_ohe(obs)


class SavedSimplePPOBot(BigTwoBot):
    def __init__(self, dir_path: str):
        self.action_cat_mapping, self.idx_cat_mapping = create_action_cat_mapping()
        self.agent = self._create_agent(dir_path)

    def action(self, observation: BigTwoObservation) -> PPOAction:
        transformed_obs = obs_to_ohe(observation)
        action_mask = generate_action_mask(self.idx_cat_mapping, observation)

        action_tensor, logp_tensor = self.agent.action(
            obs=np.array([transformed_obs]), mask=action_mask
        )
        action_cat = action_tensor.numpy()

        raw = self.action_cat_mapping[action_cat]

        cards = [c for idx, c in enumerate(observation.your_hands) if raw[idx] == 1]

        return PPOAction(
            transformed_obs=transformed_obs,
            raw_action=raw,
            action_cat=action_cat,
            action_mask=action_mask,
            logp=logp_tensor.numpy(),
            cards=cards,
        )

    def _create_agent(self, dir_path: str):
        n_action = len(self.action_cat_mapping)

        return SavedPPOAgent(
            "BigTwo",
            n_action,
            dir_path=dir_path,
        )


class SavedPastCardsPlayedBot(SavedSimplePPOBot):
    def action(self, observation: BigTwoObservation) -> PPOAction:
        temp = self.transform_obs(observation)
        transformed = {k: np.array([v]) for k, v in temp.items()}
        action_mask = generate_action_mask(self.idx_cat_mapping, observation)

        action_tensor, logp_tensor = self.agent.action(
            obs=transformed, mask=action_mask
        )
        action_cat = action_tensor.numpy()

        raw = self.action_cat_mapping[action_cat]

        cards = [c for idx, c in enumerate(observation.your_hands) if raw[idx] == 1]

        return PPOAction(
            transformed_obs=transformed,
            raw_action=raw,
            action_cat=action_cat,
            action_mask=action_mask,
            logp=logp_tensor.numpy(),
            cards=cards,
        )

    def transform_obs(self, obs: BigTwoObservation):
        seq_input = np.array(
            [cards_to_ohe(cards, 5) for cards in obs.last_n_cards_played]
        )

        if seq_input.size == 0:
            # if no carsd have been played, we will create one sequence to get the shape
            # then later on pad it out with zero.
            seq_input = np.array([cards_to_ohe([], 5)])

        # check row if we need to pad:
        if seq_input.shape[0] < 5:
            required_length = 5 - seq_input.shape[0]
            seq_input = np.pad(seq_input, [(required_length, 0), (0, 0)])

        inputs = {
            PAST_CARDS_PLAYED: seq_input,
            CURRENT_OBS: obs_to_ohe(obs, include_last_cards_played=False),
        }

        return inputs


def create_embedded_input_policy(n_action: int) -> keras.Model:
    # 52 cards + 1 empty
    card_dim = 53
    inputs = []
    embedded = []

    # current hand + current played cards (13 + 5)
    num_of_cards = 18
    for i in range(num_of_cards):
        obs_inp = layers.Input(shape=(1,), name=f"card_{i}")
        e = layers.Embedding(card_dim, 2)(obs_inp)
        e = layers.Flatten()(e)
        inputs.append(obs_inp)

        embedded.append(e)

    concat = layers.Concatenate(axis=1)(embedded)
    x = layers.Dense(n_action, activation="relu")(concat)
    output = layers.Dense(n_action)(x)
    return keras.Model(inputs=inputs, outputs=output)


def create_embedded_input_vf() -> keras.Model:
    # 52 cards + 1 empty
    card_dim = 53
    inputs = []
    embedded = []

    # current hand + current played cards (13 + 5)
    num_of_cards = 18
    for i in range(num_of_cards):
        obs_inp = layers.Input(shape=(1,), name=f"card_{i}")
        e = layers.Embedding(card_dim, 2)(obs_inp)
        e = layers.Flatten()(e)
        inputs.append(obs_inp)

        embedded.append(e)

    concat = layers.Concatenate(axis=1)(embedded)
    x = layers.Dense(256, activation="relu")(concat)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(1)(x)
    return keras.Model(inputs=inputs, outputs=output)


def lstm_policy(past_card_played_shape, obs_inp_shape, n_action: int) -> keras.Model:
    seq_inp = layers.Input(shape=past_card_played_shape, name=PAST_CARDS_PLAYED)
    x = layers.LSTM(10)(seq_inp)

    obs_inp = layers.Input(shape=obs_inp_shape, name=CURRENT_OBS)
    y = layers.Dense(64, activation="relu")(obs_inp)

    concat = layers.Concatenate(axis=1)([x, y])
    z = layers.Dense(64, activation="relu")(concat)
    output = layers.Dense(n_action)(z)
    return keras.Model(inputs=[seq_inp, obs_inp], outputs=output)


def lstm_value(past_card_played_shape, obs_inp_shape) -> keras.Model:
    seq_inp = layers.Input(shape=past_card_played_shape, name=PAST_CARDS_PLAYED)
    x = layers.LSTM(10)(seq_inp)

    obs_inp = layers.Input(shape=obs_inp_shape, name=CURRENT_OBS)
    y = layers.Dense(64, activation="relu")(obs_inp)

    concat = layers.Concatenate(axis=1)([x, y])
    z = layers.Dense(64, activation="relu")(concat)
    output = layers.Dense(1)(z)
    return keras.Model(inputs=[seq_inp, obs_inp], outputs=output)


class PastCardsPlayedBot(SimplePPOBot):
    def _create_agent(self, observation: BigTwoObservation, lr=0.0001, clip_ratio=0.3):
        n_action = len(self.action_cat_mapping)

        transformed = self.transform_obs(observation)

        past_card_played_shape = transformed[PAST_CARDS_PLAYED].shape
        obs_inp_shape = transformed[CURRENT_OBS].shape
        return PPOAgent(
            lstm_policy(
                past_card_played_shape,
                obs_inp_shape,
                n_action,
            ),
            lstm_value(past_card_played_shape, obs_inp_shape),
            "BigTwo",
            n_action,
            policy_lr=lr,
            value_lr=lr,
            clip_ratio=clip_ratio,
        )

    def action(self, observation: BigTwoObservation) -> PPOAction:
        temp = self.transform_obs(observation)
        transformed = {k: np.array([v]) for k, v in temp.items()}
        action_mask = generate_action_mask(self.idx_cat_mapping, observation)

        action_tensor, logp_tensor = self.agent.action(
            obs=transformed, mask=action_mask
        )
        action_cat = action_tensor.numpy()

        raw = self.action_cat_mapping[action_cat]

        cards = [c for idx, c in enumerate(observation.your_hands) if raw[idx] == 1]

        return PPOAction(
            transformed_obs=transformed,
            raw_action=raw,
            action_cat=action_cat,
            action_mask=action_mask,
            logp=logp_tensor.numpy(),
            cards=cards,
        )

    def transform_obs(self, obs: BigTwoObservation):
        seq_input = np.array(
            [cards_to_ohe(cards, 5) for cards in obs.last_n_cards_played]
        )

        if seq_input.size == 0:
            # if no carsd have been played, we will create one sequence to get the shape
            # then later on pad it out with zero.
            seq_input = np.array([cards_to_ohe([], 5)])

        # check row if we need to pad:
        if seq_input.shape[0] < 5:
            required_length = 5 - seq_input.shape[0]
            seq_input = np.pad(seq_input, [(required_length, 0), (0, 0)])

        inputs = {
            PAST_CARDS_PLAYED: seq_input,
            CURRENT_OBS: obs_to_ohe(obs, include_last_cards_played=False),
        }

        return inputs


class EmbeddedInputBot(SimplePPOBot):
    EMPTY_CARD_NUMBER = 52

    def _create_agent(self, observation: BigTwoObservation, lr=0.0001, clip_ratio=0.3):
        n_action = len(self.action_cat_mapping)

        return PPOAgent(
            create_embedded_input_policy(n_action),
            create_embedded_input_vf(),
            "BigTwo",
            n_action,
            policy_lr=lr,
            value_lr=lr,
            clip_ratio=clip_ratio,
        )

    def action(self, observation: BigTwoObservation) -> PPOAction:
        transformed_obs = self.transform_obs(observation)

        action_mask = generate_action_mask(self.idx_cat_mapping, observation)

        action_tensor, logp_tensor = self.agent.action(
            obs=transformed_obs, mask=action_mask
        )
        action_cat = action_tensor.numpy()

        raw = self.action_cat_mapping[action_cat]

        cards = [c for idx, c in enumerate(observation.your_hands) if raw[idx] == 1]

        return PPOAction(
            transformed_obs=transformed_obs,
            raw_action=raw,
            action_cat=action_cat,
            action_mask=action_mask,
            logp=logp_tensor.numpy(),
            cards=cards,
        )

    def transform_obs(self, obs: BigTwoObservation):
        inputs = {}
        for i in range(13):
            card_number = (
                obs.your_hands[i].to_number()
                if i < len(obs.your_hands)
                else EmbeddedInputBot.EMPTY_CARD_NUMBER
            )
            key = f"card_{i}"
            values = inputs.get(key, [])
            values.append(card_number)
            inputs[key] = values

        for i in range(5):
            card_number = (
                obs.last_cards_played[i].to_number()
                if i < len(obs.last_cards_played)
                else EmbeddedInputBot.EMPTY_CARD_NUMBER
            )

            key = f"card_{i+13}"
            values = inputs.get(key, [])
            values.append(card_number)
            inputs[key] = values

        return inputs
