import logging
import sys
import time
from collections import Counter
from typing import Dict, List

import numpy as np
import seaborn as sns
from algorithm.agent import PPOAgent, discounted_sum_of_rewards, get_mlp_vf, get_mlp_policy

from bigtwo import BigTwo
from bigtwo.bigtwo import BigTwoObservation
from playingcards.card import Card

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")


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


def save_prob_plot(obs: BigTwoObservation, action_cat_mapping, bot, filename: str) -> None:
    prob_frequency = {}
    total_prob = 0
    obs_array = obs_to_np_array(obs)
    for action_cat, raw_action in action_cat_mapping.items():
        num_of_cards = sum(raw_action)

        probs = bot.prob(obs_array, action_cat)

        prob_frequency[num_of_cards] = prob_frequency.get(num_of_cards, 0) + probs
        total_prob += probs

    ax = sns.barplot(x=list(prob_frequency.keys()), y=list(prob_frequency.values()))
    figure = ax.get_figure()
    figure.savefig(f"./plots/{filename}.png", dpi=400)


def train(batch_size=4000, epoch=50):
    train_logger = get_logger("train_ppo", log_level=logging.INFO)
    env = BigTwo()

    obs = env.reset()
    result = obs_to_np_array(obs)

    # numer of possible actions picking combination from 13 cards.
    total_num_of_actions = 2 ** 13
    action_cat_mapping = {}
    cat = 0
    for i in range(total_num_of_actions):
        action = int_to_binary_array(i)

        if sum(action) > 5:
            continue

        if sum(action) == 3 or sum(action) == 4:
            continue

        action_cat_mapping[cat] = action
        cat += 1

    bot = PPOAgent(get_mlp_policy(result.shape, len(action_cat_mapping.keys())), get_mlp_vf(result.shape), "BigTwo",
                   policy_lr=0.001, value_lr=0.001)

    episode_step = 0
    for i_episode in range(epoch):
        # make some empty lists for logging.
        batch_lens, batch_rets, batch_hands_played, batch_games_played = [], [], [], 0  # for measuring episode lengths
        env.reset()

        player_buf = create_player_buf()
        player_ep_rews = create_player_rew_buf()
        num_of_cards_played, game_history = [], []
        buf = GameBuffer()
        sample_start_time = time.time()
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
                ep_turns = env.get_hands_played()
                batch_hands_played.append(ep_turns)

                if done:
                    batch_games_played += 1
                game_history.append(env.state)

                ep_ret, ep_len = 0, 0
                # add to buf
                # process each player buf then add it to the big one
                for player_number in range(4):
                    ep_rews = player_ep_rews[player_number]
                    ep_ret += sum(ep_rews)
                    ep_len += len(ep_rews)

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

                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # reset game
                reset_obs = env.reset()
                player_buf = create_player_buf()
                player_ep_rews = create_player_rew_buf()

                if epoch_ended and i_episode % 10 == 0:
                    save_prob_plot(reset_obs, action_cat_mapping, bot, f"epoch_{i_episode}_ended_reset_probs")

        sample_time_taken = time.time() - sample_start_time

        update_start_time = time.time()
        policy_loss, value_loss = bot.update(buf)

        update_time_taken = time.time() - update_start_time

        epoch_summary = f"epoch: {i_episode + 1}, policy_loss: {policy_loss:.3f}, " \
                        f"value_loss: {value_loss:.3f}, " \
                        f"return: {np.mean(batch_rets):.3f}, " \
                        f"ep_len: {np.mean(batch_lens):.3f} " \
                        f"ep_hands_played: {np.mean(batch_hands_played):.3f} " \
                        f"ep_games_played: {batch_games_played} " \
                        f"time to sample: {sample_time_taken} " \
                        f"time to update network: {update_time_taken}"
        train_logger.info(epoch_summary)
        train_logger.info(f"num_of_cards_played_summary: {Counter(num_of_cards_played).most_common()}")

        for game in game_history[:5]:
            train_logger.info(game)


if __name__ == '__main__':
    start_time = time.time()
    train(epoch=100)
    print(f"Time taken: {time.time() - start_time:.3f} seconds")
