import logging
import sys
from collections import Counter
from typing import Dict, List
import time

import numpy as np
import scipy.signal
import scipy.signal
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers, models
from tensorflow_probability import distributions as tfd

from bigtwo import BigTwo
from bigtwo.bigtwo import BigTwoObservation
from playingcards.card import Card

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")


def discounted_sum_of_rewards(rewards, gamma):
    """
    [1 2 3]

    [
      0.99^0 * 1 + 0.99^1 * 2 + 0.99^2 * 3
      0.99^0 * 2 + 0.99^1 * 3
      0.99^0 * 3
    ]
    """
    return scipy.signal.lfilter([1], [1, float(-gamma)], rewards[::-1], axis=0)[::-1]


class PPOAgent:
    def __init__(self, obs_shape, act_dim, env_name: str, dir_path: str = None):
        self.obs_shape = obs_shape
        self.act_dim = act_dim
        self.env_name = env_name
        self.train_v_iters = 80
        self.train_p_iters = 80
        self.target_kl = 0.01
        self.clip_ratio = 0.3

        # create policy and value function from saved files.
        if dir_path is not None:
            custom_obj = {"PPO": PPO}
            self.policy = models.load_model(f"{dir_path}/{self.env_name}_policy.h5", custom_objects=custom_obj)
            self.value_function = models.load_model(f"{dir_path}/{self.env_name}_value_function.h5",
                                                    custom_objects=custom_obj)
            return

        # creating the policy
        obs_inp = layers.Input(shape=obs_shape, name="obs")
        x = layers.Dense(256, activation="relu")(obs_inp)
        x = layers.Dense(256, activation="relu")(x)
        output = layers.Dense(act_dim)(x)
        self.policy = PPO(clip_ratio=self.clip_ratio, inputs=obs_inp, outputs=output)
        self.policy.compile(optimizer=optimizers.Adam(learning_rate=0.01))

        value_input = layers.Input(shape=obs_shape, name="obs")
        x = layers.Dense(256, activation="relu")(value_input)
        x = layers.Dense(256, activation="relu")(x)
        value_output = layers.Dense(1)(x)
        self.value_function = keras.Model(inputs=value_input, outputs=value_output)
        self.value_function.compile(optimizer=optimizers.Adam(0.01), loss=keras.losses.MeanSquaredError())

    def action(self, observation):
        wrapped = np.array([observation])
        dist = tfd.Categorical(logits=self.policy(wrapped))
        sampled_action = dist.sample().numpy()
        return sampled_action[0], dist.log_prob(sampled_action).numpy()[0]

    def update(self, buf):
        obs_buf, act_buf, adv_buf, ret_buf, logp_buf = buf.get()

        # update the value function
        for _ in range(self.train_v_iters):
            value_loss = self.value_function.train_step((obs_buf, tf.constant(ret_buf)))

        # update the policy by a single step
        for i in range(self.train_p_iters):
            result = self.policy.train_step([obs_buf, act_buf, adv_buf, logp_buf])
            kl = result["approx_kl"]
            if kl > 1.5 * self.target_kl:
                print(f"Early stopping at step {i} due to reaching max kl: {kl}")
                break

        return result["loss"].numpy(), value_loss['loss'].numpy()

    def predict_value(self, observations):
        return self.value_function.predict(observations)

    def save(self, dir_path: str):
        self.policy.save(f"{dir_path}/{self.env_name}_policy.h5")
        self.value_function.save(f"{dir_path}/{self.env_name}_value_function.h5")


class PPO(keras.Model):
    def __init__(self, clip_ratio=0.2, **kwargs):
        self.clip_ratio = clip_ratio
        super(PPO, self).__init__(**kwargs)

    def train_step(self, data):
        batch_obs, batch_acts, batch_adv, previous_log_prob = data

        with tf.GradientTape() as tape:
            result = self(batch_obs, training=True)
            curr_log_probs = tfd.Categorical(logits=result).log_prob(batch_acts)

            policy_ratio = tf.exp(curr_log_probs - previous_log_prob) * batch_adv
            epsilon = tf.constant(self.clip_ratio)
            limit = tf.where(batch_adv > 0, (1 + epsilon) * batch_adv, (1 - epsilon) * batch_adv)
            policy_changes = tf.math.minimum(policy_ratio, limit)

            approx_kl = tf.reduce_mean(previous_log_prob - curr_log_probs)

            loss = -tf.reduce_mean(policy_changes)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradient = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradient, trainable_vars))

        return {"loss": loss, "approx_kl": approx_kl}


def obs_to_np_array(obs: BigTwoObservation) -> np.ndarray:
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


class PlayerBuffer:
    def __init__(self, gamma=0.99, lam=0.95):
        self.obs_buf = []
        self.act_buf = []
        self.adv_buf = []
        self.rew_buf = []
        self.ret_buf = []
        self.logp_buf = []
        self.gamma, self.lam = gamma, lam

    def store(self, obs, act, rew, logp):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.logp_buf.append(logp)

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


class GameBuffer:
    def __init__(self):
        self.obs_buf = None
        self.act_buf = None
        self.adv_buf = None
        self.rew_buf = None
        self.ret_buf = None
        self.logp_buf = None

    def add(self, pbuf: PlayerBuffer):
        self.obs_buf = np.array(pbuf.obs_buf) if self.obs_buf is None else np.append(self.obs_buf, pbuf.obs_buf, axis=0)
        self.act_buf = np.array(pbuf.act_buf) if self.act_buf is None else np.append(self.act_buf, pbuf.act_buf, axis=0)
        self.adv_buf = pbuf.adv_buf if self.adv_buf is None else np.append(self.adv_buf, pbuf.adv_buf, axis=0)
        self.rew_buf = np.array(pbuf.rew_buf) if self.rew_buf is None else np.append(self.rew_buf, pbuf.rew_buf, axis=0)
        self.ret_buf = pbuf.ret_buf if self.ret_buf is None else np.append(self.ret_buf, pbuf.ret_buf, axis=0)
        self.logp_buf = np.array(pbuf.logp_buf) if self.logp_buf is None else np.append(self.logp_buf, pbuf.logp_buf,
                                                                                        axis=0)

    def get(self):
        return self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf


def create_player_buf(num_of_player=4) -> Dict[int, PlayerBuffer]:
    return {i: PlayerBuffer() for i in range(num_of_player)}


def create_player_rew_buf(num_of_player=4) -> Dict[int, List[int]]:
    return {i: [] for i in range(num_of_player)}


def int_to_binary_array(v: int) -> List[int]:
    result = []
    for c in '{0:013b}'.format(v):
        result.append(int(c))
    return result


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_logger(logger_name, log_level):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.addHandler(get_console_handler())
    return logger


def train(batch_size=8000, epoch=50):
    train_logger = get_logger("train_ppo", log_level=logging.INFO)
    env = BigTwo()

    obs = env.reset()
    result = obs_to_np_array(obs)

    # numer of possible actions picking combination from 13 cards.
    num_of_actions = 2 ** 13

    action_cat_mapping = {i: int_to_binary_array(i) for i in range(num_of_actions)}

    bot = PPOAgent(result.shape, num_of_actions, "BigTwo")
    episode_step = 0
    for i_episode in range(epoch):
        # make some empty lists for logging.
        batch_lens = []  # for measuring episode lengths
        batch_rets = []
        batch_hands_played = []
        batch_games_played = 0
        env.reset()

        player_buf = create_player_buf()
        player_ep_rews = create_player_rew_buf()
        num_of_cards_played = []
        buf = GameBuffer()
        for t in range(batch_size):
            obs = env.get_current_player_obs()

            before_play_summary = f"turn: {episode_step}" \
                                  f"current_player: {obs.current_player}" \
                                  f"before player hand: {obs.your_hands}" \
                                  f"last_player_played: {obs.last_player_played}" \
                                  f"cards played: {obs.last_cards_played}"
            train_logger.debug(before_play_summary)

            obs_array = obs_to_np_array(obs)
            action_cat, logp = bot.action(obs_array)
            action = action_cat_mapping[action_cat]

            num_of_cards = sum(action)
            num_of_cards_played.append(num_of_cards)
            new_obs, reward, done = env.step(action)

            player_ep_rews[obs.current_player].append(reward)

            # storing the trajectory per players because the awards is per player not sum of all players.
            player_buf[obs.current_player].store(obs_array, action_cat, reward, logp)
            episode_step += 1

            after_play_summary = f"action: {action}" \
                                 f"after player hand: {new_obs.your_hands}" \
                                 f"===="
            train_logger.debug(after_play_summary)

            epoch_ended = t == batch_size - 1
            if done or epoch_ended:
                # use the game winning path to log
                ep_rews = player_ep_rews[obs.current_player]
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                ep_turns = env.get_hands_played()
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                batch_hands_played.append(ep_turns)

                if done:
                    batch_games_played += 1

                # add to buf
                # process each player buf then add it to the big one
                for player_number in range(4):
                    if player_buf[player_number].is_empty():
                        continue

                    estimated_values = bot.predict_value(player_buf[player_number].get_curr_path())
                    last_val = 0
                    if not done and epoch_ended:
                        last_obs = env.get_player_obs(player_number)
                        last_obs_arr = np.array([obs_to_np_array(last_obs)])
                        last_val = bot.predict_value(last_obs_arr)
                    player_buf[player_number].finish_path(estimated_values, last_val)
                    buf.add(player_buf[player_number])

                # reset game
                env.reset()
                player_buf = create_player_buf()
                player_ep_rews = create_player_rew_buf()

        policy_loss, value_loss = bot.update(buf)

        total_rewards_mean = np.mean(batch_rets)
        epoch_summary = f"epoch: {i_episode + 1}, policy_loss: {policy_loss:.3f}, " \
                        f"value_loss: {value_loss:.3f}, " \
                        f"return: {total_rewards_mean:.3f}, " \
                        f"ep_len: {np.mean(batch_lens):.3f} " \
                        f"ep_hands_played: {np.mean(batch_hands_played):.3f} " \
                        f"ep_games_played: {batch_games_played}"
        train_logger.info(epoch_summary)
        train_logger.info(f"num_of_cards_played_summary: {Counter(num_of_cards_played).most_common()}")


if __name__ == '__main__':
    start_time = time.time()
    train()
    print(f"Time taken: {time.time() - start_time:.3f} seconds")
