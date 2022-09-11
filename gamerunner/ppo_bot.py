from typing import Iterable, Mapping, Optional, Union

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers

from algorithm.agent import (
    PPOAgent,
    PPOBufferInterface,
    SavedPPOAgent,
    discounted_sum_of_rewards,
    get_mlp_vf,
)
from bigtwo.bigtwo import BigTwoObservation
from bigtwo.preprocessing import create_action_cat_mapping, obs_to_ohe, cards_to_ohe, generate_action_mask
from gamerunner.big_two_bot import BigTwoBot

PAST_CARDS_PLAYED = "past_cards_played"
CURRENT_OBS = "current_obs"


def set_or_add(
    original: Optional[np.ndarray], new: Union[np.ndarray, Iterable]
) -> np.ndarray:
    if original is None:
        return np.array(new)

    return np.append(original, new, axis=0)


class PlayerBuffer:
    def __init__(self, gamma=0.99, lam=0.95):
        self.obs_buf = None
        self.act_buf = []
        self.adv_buf = []
        self.rew_buf = []
        self.ret_buf = []
        self.logp_buf = []
        self.mask_buf = []
        self.gamma, self.lam = gamma, lam

    def store(self, obs, act, rew, logp, mask):
        self.obs_buf = set_or_add(self.obs_buf, obs)
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


def create_ppo_action(
    agent,
    agent_input,
    action_mask,
    action_cat_mapping,
    obs: BigTwoObservation,
) -> PPOAction:
    action_tensor, logp_tensor = agent.action(obs=agent_input, mask=action_mask)
    action_cat = action_tensor.numpy()

    raw = action_cat_mapping[action_cat]

    cards = [c for idx, c in enumerate(obs.your_hands) if raw[idx] == 1]

    return PPOAction(
        transformed_obs=agent_input,
        raw_action=raw,
        action_cat=action_cat,
        action_mask=action_mask,
        logp=logp_tensor.numpy(),
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

    def action(self, obs: BigTwoObservation) -> PPOAction:
        transformed_obs = self.transform_obs(obs)
        action_mask = generate_action_mask(self.idx_cat_mapping, obs)
        agent_input = np.array([transformed_obs])

        return create_ppo_action(
            self.agent, agent_input, action_mask, self.action_cat_mapping, obs
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

    def action(self, obs: BigTwoObservation) -> PPOAction:
        transformed_obs = obs_to_ohe(obs)
        action_mask = generate_action_mask(self.idx_cat_mapping, obs)
        agent_input = np.array([transformed_obs])

        return create_ppo_action(
            self.agent, agent_input, action_mask, self.action_cat_mapping, obs
        )

    def _create_agent(self, dir_path: str):
        n_action = len(self.action_cat_mapping)

        return SavedPPOAgent(
            "BigTwo",
            n_action,
            dir_path=dir_path,
        )


def obs_to_past_cards_played_input(obs: BigTwoObservation) -> Mapping[str, np.ndarray]:
    seq_input = np.array([cards_to_ohe(cards, 5) for cards in obs.last_n_cards_played])

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


class SavedPastCardsPlayedBot(SavedSimplePPOBot):
    def action(self, obs: BigTwoObservation) -> PPOAction:
        temp = obs_to_past_cards_played_input(obs)
        agent_input = {k: np.array([v]) for k, v in temp.items()}
        action_mask = generate_action_mask(self.idx_cat_mapping, obs)

        return create_ppo_action(
            self.agent, agent_input, action_mask, self.action_cat_mapping, obs
        )


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

    def action(self, obs: BigTwoObservation) -> PPOAction:
        temp = self.transform_obs(obs)
        agent_input = {k: np.array([v]) for k, v in temp.items()}
        action_mask = generate_action_mask(self.idx_cat_mapping, obs)
        return create_ppo_action(
            self.agent, agent_input, action_mask, self.action_cat_mapping, obs
        )

    def transform_obs(self, obs: BigTwoObservation):
        return obs_to_past_cards_played_input(obs)


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

    def action(self, obs: BigTwoObservation) -> PPOAction:
        transformed_obs = self.transform_obs(obs)

        action_mask = generate_action_mask(self.idx_cat_mapping, obs)

        action_tensor, logp_tensor = self.agent.action(
            obs=transformed_obs, mask=action_mask
        )
        action_cat = action_tensor.numpy()

        raw = self.action_cat_mapping[action_cat]

        cards = [c for idx, c in enumerate(obs.your_hands) if raw[idx] == 1]

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
