import itertools
import logging
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import seaborn as sns
from algorithm.agent import PPOAgent, discounted_sum_of_rewards, get_mlp_vf, get_mlp_policy

from bigtwo.bigtwo import BigTwoObservation, BigTwo, BigTwoHand
from playingcards.card import Card, Suit, Rank

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")


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


class GameBuffer:
    def __init__(self):
        self.obs_buf = None
        self.act_buf = None
        self.adv_buf = None
        self.rew_buf = None
        self.ret_buf = None
        self.logp_buf = None
        self.mask_buf = None

    def add(self, pbuf: PlayerBuffer):
        self.obs_buf = np.array(pbuf.obs_buf) if self.obs_buf is None else np.append(self.obs_buf, pbuf.obs_buf, axis=0)
        self.act_buf = np.array(pbuf.act_buf) if self.act_buf is None else np.append(self.act_buf, pbuf.act_buf, axis=0)
        self.adv_buf = pbuf.adv_buf if self.adv_buf is None else np.append(self.adv_buf, pbuf.adv_buf, axis=0)
        self.rew_buf = np.array(pbuf.rew_buf) if self.rew_buf is None else np.append(self.rew_buf, pbuf.rew_buf, axis=0)
        self.ret_buf = pbuf.ret_buf if self.ret_buf is None else np.append(self.ret_buf, pbuf.ret_buf, axis=0)
        self.logp_buf = np.array(pbuf.logp_buf) if self.logp_buf is None else np.append(self.logp_buf, pbuf.logp_buf,
                                                                                        axis=0)
        self.mask_buf = np.array(pbuf.mask_buf) if self.mask_buf is None else np.append(self.mask_buf, pbuf.mask_buf,
                                                                                        axis=0)

    def get(self):
        return self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf


class SampleMetric:
    def __init__(self):
        self.events_counter = Counter({})
        self.batch_lens = []
        self.batch_rets = []
        self.batch_hands_played = []
        self.batch_games_played = 0
        self.num_of_cards_played = []
        self.game_history = []

    def track_action(self, action):
        num_of_cards = sum(action)
        self.num_of_cards_played.append(num_of_cards)

    def track_env(self, env: BigTwo, done=False):
        ep_turns = env.get_hands_played()
        self.batch_hands_played.append(ep_turns)

        if done:
            self.batch_games_played += 1
        self.game_history.append(env.state)

        self.events_counter += Counter(env.event_count)

    def track_rewards(self, ep_ret, ep_len):
        self.batch_rets.append(ep_ret)
        self.batch_lens.append(ep_len)


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


def obs_to_ohe_np_array(obs: BigTwoObservation) -> np.ndarray:
    suit_ohe = {
        Suit.spades: [0, 0, 0, 1],
        Suit.hearts: [0, 0, 1, 0],
        Suit.clubs: [0, 1, 0, 0],
        Suit.diamond: [1, 0, 0, 0]
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

    data = []
    data += obs.num_card_per_player

    for i in range(5):
        # add place holder for empty card
        if i >= len(obs.last_cards_played):
            data += empty_suit
            data += empty_rank
            continue

        curr_card = obs.last_cards_played[i]
        data += suit_ohe[curr_card.suit]
        data += rank_ohe[curr_card.rank]

    for i in range(13):
        if i >= len(obs.your_hands):
            data += empty_suit
            data += empty_rank
            continue

        curr_card = obs.your_hands[i]
        data += suit_ohe[curr_card.suit]
        data += rank_ohe[curr_card.rank]

    return np.array(data)


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


def save_ep_returns_plot(values) -> None:
    ax = sns.lineplot(data=values)
    figure = ax.get_figure()
    figure.savefig(f"./plots/ppo_runner_ep_returns.png", dpi=400)


def create_action_cat_mapping(num_cards_in_hand=13) -> Tuple[Dict[int, List[int]], Dict[frozenset, int]]:
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


def generate_action_mask(act_cat_mapping: Dict[int, List[int]], idx_cat_mapping, player_hand: BigTwoHand) -> np.array:
    result = np.full(len(act_cat_mapping), False)

    result[0] = True

    card_idx_mapping = {}
    for idx in range(len(player_hand)):
        cat = idx_cat_mapping[frozenset([idx])]
        card_idx_mapping[player_hand[idx]] = idx
        result[cat] = True

    if len(player_hand) < 2:
        return result

    for pair in player_hand.pairs:
        pair_idx = []
        for card in pair:
            pair_idx.append(card_idx_mapping.get(card))

        cat = idx_cat_mapping[frozenset(pair_idx)]
        result[cat] = True

    if len(player_hand) < 5:
        return result

    for _, combinations in player_hand.combinations.items():
        for comb in combinations:
            comb_idx = []
            for card in comb:
                comb_idx.append(card_idx_mapping.get(card))

            cat = idx_cat_mapping[frozenset(comb_idx)]
            result[cat] = True

    return result


def collect_data_from_env(bot, buffer_size=4000) -> Tuple[GameBuffer, SampleMetric]:
    env = BigTwo()

    # numer of possible actions picking combination from 13 cards.
    action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()

    player_buf = create_player_buf()
    player_ep_rews = create_player_rew_buf()

    sample_metric = SampleMetric()

    buf = GameBuffer()

    for t in range(buffer_size):
        obs = env.get_current_player_obs()
        obs_array = obs_to_ohe_np_array(obs)

        action_mask = generate_action_mask(action_cat_mapping, idx_cat_mapping, obs.your_hands)
        action_cat, logp = bot.action(obs_array, action_mask)
        action = action_cat_mapping[action_cat]

        sample_metric.track_action(action)

        new_obs, reward, done = env.step(action)

        # storing the trajectory per players because the awards is per player not sum of all players.
        player_ep_rews[obs.current_player].append(reward)
        player_buf[obs.current_player].store(obs_array, action_cat, reward, logp, action_mask)

        epoch_ended = t == buffer_size - 1
        if done or epoch_ended:
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
                    last_obs_arr = np.array([obs_to_ohe_np_array(last_obs)])
                    last_val = bot.predict_value(last_obs_arr)
                player_buf[player_number].finish_path(estimated_values, last_val)
                buf.add(player_buf[player_number])

            sample_metric.track_rewards(ep_ret, ep_len)
            sample_metric.track_env(env, done)

            # reset game
            env.reset()
            player_buf = create_player_buf()
            player_ep_rews = create_player_rew_buf()

    return buf, sample_metric


def train_serialise(epoch=50):
    train_logger = get_logger("train_ppo", log_level=logging.INFO)
    env = BigTwo()

    obs = env.reset()
    result = obs_to_ohe_np_array(obs)

    # numer of possible actions picking combination from 13 cards.
    action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()

    lr = 0.001
    bot = PPOAgent(get_mlp_policy(result.shape, len(action_cat_mapping)), get_mlp_vf(result.shape), "BigTwo",
                   policy_lr=lr, value_lr=lr)

    ep_returns = []
    for i_episode in range(epoch):
        sample_start_time = time.time()
        buf, m = collect_data_from_env(bot)

        sample_time_taken = time.time() - sample_start_time

        update_start_time = time.time()
        policy_loss, value_loss = bot.update(buf, mini_batch_size=512)

        update_time_taken = time.time() - update_start_time

        epoch_summary = f"epoch: {i_episode + 1}, policy_loss: {policy_loss:.3f}, " \
                        f"value_loss: {value_loss:.3f}, " \
                        f"return: {np.mean(m.batch_rets):.3f}, " \
                        f"ep_len: {np.mean(m.batch_lens):.3f} " \
                        f"ep_hands_played: {np.mean(m.batch_hands_played):.3f} " \
                        f"ep_games_played: {m.batch_games_played} " \
                        f"time to sample: {sample_time_taken} " \
                        f"time to update network: {update_time_taken}"
        train_logger.info(epoch_summary)
        train_logger.info(f"num_of_cards_played_summary: {Counter(m.num_of_cards_played).most_common()}")

        ep_returns.append(np.mean(m.batch_rets))
        for game in m.game_history[:5]:
            train_logger.info(game)

        train_logger.info(f"event counter: {m.events_counter}")

    # save_ep_returns_plot(ep_returns)


if __name__ == '__main__':
    train_serialise(epoch=1)
    # print(f"Time taken: {time.time() - start_time:.3f} seconds")
