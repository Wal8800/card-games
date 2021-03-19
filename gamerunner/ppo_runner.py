import itertools
import logging
import multiprocessing as mp
import sys
import time
from collections import Counter
from multiprocessing import Pool, Queue, Process
from typing import Dict, List, Tuple

import numpy as np
import seaborn as sns
import tensorflow as tf
from algorithm.agent import (
    PPOAgent,
    discounted_sum_of_rewards,
    get_mlp_vf,
    get_mlp_policy,
)

from bigtwo.bigtwo import BigTwoObservation, BigTwo, BigTwoHand
from gamerunner.cmd_line_bot import CommandLineBot
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
        self.obs_buf = (
            np.array(pbuf.obs_buf)
            if self.obs_buf is None
            else np.append(self.obs_buf, pbuf.obs_buf, axis=0)
        )
        self.act_buf = (
            np.array(pbuf.act_buf)
            if self.act_buf is None
            else np.append(self.act_buf, pbuf.act_buf, axis=0)
        )
        self.adv_buf = (
            pbuf.adv_buf
            if self.adv_buf is None
            else np.append(self.adv_buf, pbuf.adv_buf, axis=0)
        )
        self.rew_buf = (
            np.array(pbuf.rew_buf)
            if self.rew_buf is None
            else np.append(self.rew_buf, pbuf.rew_buf, axis=0)
        )
        self.ret_buf = (
            pbuf.ret_buf
            if self.ret_buf is None
            else np.append(self.ret_buf, pbuf.ret_buf, axis=0)
        )
        self.logp_buf = (
            np.array(pbuf.logp_buf)
            if self.logp_buf is None
            else np.append(self.logp_buf, pbuf.logp_buf, axis=0)
        )
        self.mask_buf = (
            np.array(pbuf.mask_buf)
            if self.mask_buf is None
            else np.append(self.mask_buf, pbuf.mask_buf, axis=0)
        )

    def get(self):
        return (
            self.obs_buf,
            self.act_buf,
            self.adv_buf.astype("float32"),
            self.ret_buf.astype("float32"),
            self.logp_buf,
        )


class SampleMetric:
    def __init__(self):
        self.events_counter = Counter({})
        self.batch_lens = []
        self.batch_rets = []
        self.batch_hands_played = []
        self.num_of_cards_played = []
        self.num_of_valid_card_played = []
        self.game_history = []
        self.batch_games_played = 0

    def track_action(self, action, rewards):
        num_of_cards = sum(action)
        if rewards >= 0:
            self.num_of_valid_card_played.append(num_of_cards)

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


def config_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


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
    for c in "{0:013b}".format(v):
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


def save_ep_returns_plot(values, epoch, name="ep_ret") -> None:
    ax = sns.lineplot(data=values)
    figure = ax.get_figure()
    figure.savefig(f"./plots/ppo_{name}_{int(time.time())}_{epoch}.png", dpi=400)


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


def generate_action_mask(
    act_cat_mapping: Dict[int, List[int]],
    idx_cat_mapping,
    obs: BigTwoObservation,
) -> np.array:
    result = np.full(len(act_cat_mapping), False)

    result[0] = obs.can_skip()

    card_idx_mapping = {}
    for idx in range(len(obs.your_hands)):
        cat = idx_cat_mapping[frozenset([idx])]
        card_idx_mapping[obs.your_hands[idx]] = idx
        result[cat] = True

    if len(obs.your_hands) < 2:
        return result

    for pair in obs.your_hands.pairs:
        pair_idx = []
        for card in pair:
            pair_idx.append(card_idx_mapping.get(card))

        cat = idx_cat_mapping[frozenset(pair_idx)]
        result[cat] = True

    if len(obs.your_hands) < 5:
        return result

    for _, combinations in obs.your_hands.combinations.items():
        for comb in combinations:
            comb_idx = []
            for card in comb:
                comb_idx.append(card_idx_mapping.get(card))

            cat = idx_cat_mapping[frozenset(comb_idx)]
            result[cat] = True

    return result


def sample_worker(input: Queue, output: Queue):
    config_gpu()

    for policy_weight, value_weight, buffer_size in iter(input.get, "STOP"):
        buf, m = collect_data_from_env(policy_weight, value_weight, buffer_size)
        output.put((buf, m))


def create_ppo_agent(obs_shape, n_action: int, lr: float = 0.01) -> PPOAgent:
    return PPOAgent(
        get_mlp_policy(obs_shape, n_action, hidden_units=512),
        get_mlp_vf(obs_shape, hidden_units=512),
        "BigTwo",
        n_action,
        policy_lr=lr,
        value_lr=lr,
    )


def collect_data_from_env(
    policy_weight, value_weight, buffer_size=4000
) -> Tuple[GameBuffer, SampleMetric]:
    env = BigTwo()

    obs = env.reset()
    result = obs_to_ohe_np_array(obs)

    # numer of possible actions picking combination from 13 cards.
    action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()
    n_action = len(action_cat_mapping)
    bot = create_ppo_agent(result.shape, n_action)
    bot.set_weights(policy_weight, value_weight)

    player_buf = create_player_buf()
    player_ep_rews = create_player_rew_buf()

    sample_metric = SampleMetric()

    buf = GameBuffer()

    for t in range(buffer_size):
        obs = env.get_current_player_obs()
        obs_array = obs_to_ohe_np_array(obs)

        action_mask = generate_action_mask(action_cat_mapping, idx_cat_mapping, obs)

        action_tensor, logp_tensor = bot.action(obs=obs_array, mask=action_mask)
        action_cat = action_tensor.numpy()
        logp = logp_tensor.numpy()

        action = action_cat_mapping[action_cat]

        new_obs, reward, done = env.step(action)
        sample_metric.track_action(action, reward)

        # storing the trajectory per players because the awards is per player not sum of all players.
        player_ep_rews[obs.current_player].append(reward)
        player_buf[obs.current_player].store(
            obs_array, action_cat, reward, logp, action_mask
        )

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

                estimated_values = bot.predict_value(
                    player_buf[player_number].get_curr_path()
                )
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


def create_worker_task(num_cpu, policy_weight, value_weight, buffer_size):
    chunk_buffer_size = buffer_size // num_cpu

    args = zip(
        itertools.repeat(policy_weight, num_cpu),
        itertools.repeat(value_weight, num_cpu),
        itertools.repeat(chunk_buffer_size, num_cpu),
    )

    return args


def collect_data_parallel(
    policy_weight, value_weight, buffer_size=4000
) -> Tuple[GameBuffer, SampleMetric]:
    num_cpu = 4

    chunk_buffer_size = buffer_size // num_cpu

    args = list(
        zip(
            itertools.repeat(policy_weight, num_cpu),
            itertools.repeat(value_weight, num_cpu),
            itertools.repeat(chunk_buffer_size, num_cpu),
        )
    )

    with Pool(processes=4) as pool:
        result: List[Tuple[GameBuffer, SampleMetric]] = pool.starmap(
            collect_data_from_env, args
        )

    return result[0][0], result[0][1]


def train(epoch=50):
    train_logger = get_logger("train_ppo", log_level=logging.INFO)
    env = BigTwo()

    obs = env.reset()
    result = obs_to_ohe_np_array(obs)

    # numer of possible actions picking combination from 13 cards.
    action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()
    n_action = len(action_cat_mapping)
    lr = 0.001
    bot = PPOAgent(
        get_mlp_policy(result.shape, n_action),
        get_mlp_vf(result.shape),
        "BigTwo",
        n_action,
        policy_lr=lr,
        value_lr=lr,
    )

    ep_returns = []
    for i_episode in range(epoch):
        sample_start_time = time.time()

        policy_weight, value_weight = bot.get_weights()

        buf, m = collect_data_from_env(policy_weight, value_weight)

        sample_time_taken = time.time() - sample_start_time

        update_start_time = time.time()
        policy_loss, value_loss = bot.update(buf, mini_batch_size=512)

        update_time_taken = time.time() - update_start_time

        epoch_summary = (
            f"epoch: {i_episode + 1}, policy_loss: {policy_loss:.3f}, "
            f"value_loss: {value_loss:.3f}, "
            f"return: {np.mean(m.batch_rets):.3f}, "
            f"ep_len: {np.mean(m.batch_lens):.3f} "
            f"ep_hands_played: {np.mean(m.batch_hands_played):.3f} "
            f"ep_games_played: {m.batch_games_played} "
            f"time to sample: {sample_time_taken} "
            f"time to update network: {update_time_taken}"
        )
        train_logger.info(epoch_summary)
        train_logger.info(
            f"num_of_cards_played_summary: {Counter(m.num_of_cards_played).most_common()}"
        )

        ep_returns.append(np.mean(m.batch_rets))
        for game in m.game_history[:5]:
            train_logger.info(game)

        train_logger.info(f"event counter: {m.events_counter}")

    # save_ep_returns_plot(ep_returns)


def merge_result(
    results: List[Tuple[GameBuffer, SampleMetric]]
) -> Tuple[GameBuffer, SampleMetric]:
    result_buf, result_metric = None, None

    for buf, m in results:
        if result_buf is None:
            result_buf = buf
            result_metric = m
            continue

        result_buf.obs_buf = np.append(result_buf.obs_buf, buf.obs_buf, axis=0)
        result_buf.act_buf = np.append(result_buf.act_buf, buf.act_buf, axis=0)
        result_buf.adv_buf = np.append(result_buf.adv_buf, buf.adv_buf, axis=0)
        result_buf.rew_buf = np.append(result_buf.rew_buf, buf.rew_buf, axis=0)
        result_buf.ret_buf = np.append(result_buf.ret_buf, buf.ret_buf, axis=0)
        result_buf.logp_buf = np.append(result_buf.logp_buf, buf.logp_buf, axis=0)
        result_buf.mask_buf = np.append(result_buf.mask_buf, buf.mask_buf, axis=0)

        result_metric.events_counter += m.events_counter
        result_metric.batch_lens += m.batch_lens
        result_metric.batch_rets += m.batch_rets
        result_metric.batch_hands_played += m.batch_hands_played
        result_metric.num_of_cards_played += m.num_of_cards_played
        result_metric.game_history += [h for h in m.game_history if len(h) > 0]
        result_metric.batch_games_played += m.batch_games_played
        result_metric.num_of_valid_card_played += m.num_of_valid_card_played

    return result_buf, result_metric


def train_parallel(epoch=50, buffer_size=4000, lr=0.001):
    mp.set_start_method("spawn")

    # setting up worker for sampling
    NUMBER_OF_PROCESSES = 10
    task_queue = Queue()
    done_queue = Queue()

    # Start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        Process(target=sample_worker, args=(task_queue, done_queue)).start()

    try:
        train_logger = get_logger("train_ppo", log_level=logging.INFO)
        env = BigTwo()
        obs = env.reset()
        obs_array = obs_to_ohe_np_array(obs)

        # numer of possible actions picking combination from 13 cards.
        action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()
        n_action = len(action_cat_mapping)

        bot = create_ppo_agent(obs_array.shape, n_action, lr)

        ep_returns = []
        min_ep_returns = []
        for i_episode in range(epoch):
            sample_start_time = time.time()

            policy_weight, value_weight = bot.get_weights()

            tasks = create_worker_task(
                NUMBER_OF_PROCESSES, policy_weight, value_weight, buffer_size
            )
            for task in tasks:
                task_queue.put(task)

            result: List[Tuple[GameBuffer, SampleMetric]] = []
            for i in range(NUMBER_OF_PROCESSES):
                result.append(done_queue.get())

            buf, m = merge_result(result)
            sample_time_taken = time.time() - sample_start_time

            update_start_time = time.time()
            policy_loss, value_loss = bot.update(buf, mini_batch_size=512)

            update_time_taken = time.time() - update_start_time

            epoch_summary = (
                f"epoch: {i_episode + 1}, policy_loss: {policy_loss:.3f}, "
                f"value_loss: {value_loss:.3f}, "
                f"return: {np.mean(m.batch_rets):.3f}, "
                f"min returns: {np.min(m.batch_rets)}, "
                f"ep_len: {np.mean(m.batch_lens):.3f}, "
                f"ep_hands_played: {np.mean(m.batch_hands_played):.3f} "
                f"ep_games_played: {m.batch_games_played} "
                f"time to sample: {sample_time_taken} "
                f"time to update network: {update_time_taken}"
            )
            train_logger.info(epoch_summary)

            ep_returns.append(np.mean(m.batch_rets))
            min_ep_returns.append(np.min(m.batch_rets))
            train_logger.info(
                f"num_of_cards_played_summary: {Counter(m.num_of_cards_played).most_common()}, "
                f"num_of_valid_cards_played_summary: {Counter(m.num_of_valid_card_played).most_common()}"
            )

            train_logger.info(f"event counter: {m.events_counter}")

        save_ep_returns_plot(min_ep_returns, epoch, name="min_ep_returns")
        save_ep_returns_plot(ep_returns, epoch)
        bot.save("save")
    finally:
        # Tell child processes to stop
        for i in range(NUMBER_OF_PROCESSES):
            task_queue.put("STOP")


def play_with_cmd():
    env = BigTwo()

    obs = env.reset()
    result = obs_to_ohe_np_array(obs)

    # numer of possible actions picking combination from 13 cards.
    action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()
    n_action = len(action_cat_mapping)

    player_list = []
    for i in range(BigTwo.number_of_players() - 1):
        bot = PPOAgent(
            get_mlp_policy(result.shape, n_action),
            get_mlp_vf(result.shape),
            "BigTwo",
            n_action,
        )

        player_list.append(bot)

    player_list.append(CommandLineBot())

    episode_step = 0
    done = False
    while not done:
        obs = env.get_current_player_obs()
        print("turn ", episode_step)
        print("current_player", obs.current_player)
        print(f"before player hand: {obs.your_hands}")
        print("last_player_played: ", obs.last_player_played)
        print(f"cards played: {obs.last_cards_played}")

        current_player = player_list[obs.current_player]

        if isinstance(current_player, CommandLineBot):
            action = current_player.action(obs)
            new_obs, reward, done = env.step(action)
            print(f"cmd bot reward: {reward}")
        else:
            action_mask = generate_action_mask(action_cat_mapping, idx_cat_mapping, obs)

            obs_array = obs_to_ohe_np_array(obs)
            action_tensor, _ = current_player.action(obs=obs_array, mask=action_mask)
            action_cat = action_tensor.numpy()

            action = action_cat_mapping[action_cat]
            new_obs, reward, done = env.step(action)

        episode_step += 1
        print("action: " + str(action))
        print(f"after player hand: {new_obs.your_hands}")
        print(env.event_count)
        print("====")

    env.display_all_player_hands()


if __name__ == "__main__":
    config_gpu()
    start_time = time.time()
    train_parallel(epoch=2000)
    # play_with_cmd()
    print(f"Time taken: {time.time() - start_time:.3f} seconds")
