import cProfile
import copy
import gc
import io
import itertools
import linecache
import logging
import multiprocessing as mp
import os
import pickle
import pstats
import random
import sys
import time
import tracemalloc
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from multiprocessing import Queue, Process
from pstats import SortKey
from typing import Dict, List, Tuple, Any, Mapping

import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
from pympler import summary, muppy

from algorithm.agent import PPOBufferInterface
from bigtwo.bigtwo import BigTwo, BigTwoObservation, BigTwoHand
from gamerunner.ppo_bot import (
    SimplePPOBot,
    GameBuffer,
    PlayerBuffer,
    SavedSimplePPOBot,
    MultiInputGameBuffer,
    MultiInputPlayerBuffer,
    PastCardsPlayedBot,
    SavedPastCardsPlayedBot,
)
from playingcards.card import Card

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")


def config_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_logger(logger_name, log_level):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.addHandler(get_console_handler())
    return logger


def create_player_buf(player_buf_class) -> Dict[int, Any]:
    return {i: player_buf_class() for i in range(4)}


def create_player_rew_buf(num_of_player=4) -> Dict[int, List[int]]:
    return {i: [] for i in range(num_of_player)}


class BotType(Enum):
    SIMPLE_PPO_BOT = "SimplePPOBot"
    LSTM_PPO_BOT = "LSTMPPOBot"


def bot_type_from_str(value: str) -> BotType:
    for enum_value in BotType:
        if enum_value.value == value:
            return enum_value

    raise ValueError(f"Input value ({value}) doesn't match with any bot type enum")


class BotBuilder:
    @staticmethod
    def create_training_bot(bot_type: BotType, *args, **kwargs):
        if bot_type == BotType.SIMPLE_PPO_BOT:
            return SimplePPOBot(*args, **kwargs)

        if bot_type == BotType.LSTM_PPO_BOT:
            return PastCardsPlayedBot(*args, **kwargs)

        raise ValueError("Unexpected bot type")

    @staticmethod
    def create_testing_bot(bot_type: BotType, *args, **kwargs):
        if bot_type == BotType.SIMPLE_PPO_BOT:
            return SavedSimplePPOBot(*args, **kwargs)

        if bot_type == BotType.LSTM_PPO_BOT:
            return SavedPastCardsPlayedBot(*args, **kwargs)

        raise ValueError("Unexpected bot type")

    @staticmethod
    def create_testing_bot_by_str(value: str, *args, **kwargs):
        bot_type = bot_type_from_str(value)

        return BotBuilder.create_testing_bot(bot_type, *args, **kwargs)


def get_game_buf(bot_type: BotType) -> PPOBufferInterface:
    if bot_type == BotType.SIMPLE_PPO_BOT:
        return GameBuffer()

    if bot_type == BotType.LSTM_PPO_BOT:
        return MultiInputGameBuffer()

    raise ValueError("unexpected bot type")


def get_player_buf(bot_type: BotType):
    if bot_type == BotType.SIMPLE_PPO_BOT:
        return PlayerBuffer()

    if bot_type == BotType.LSTM_PPO_BOT:
        return MultiInputPlayerBuffer()

    raise ValueError("unexpected bot type")


@dataclass
class ExperimentConfig:
    epoch: int = 20000
    lr: float = 0.0001
    buffer_size: int = 4000
    mini_batch_size: int = 512

    num_of_worker: int = 10

    clip_ratio: float = 0.3

    opponent_buf_limit = 10
    opponent_update_freq = 100

    bot_type = BotType.SIMPLE_PPO_BOT


class SampleMetric:
    """
    Track the metric for player 0 when playing big two
    """

    def __init__(self):
        self.batch_lens = []
        self.batch_rets = []
        self.batch_hands_played = []
        self.batch_games_played: int = 0
        self.win_count = []
        self.played_first_turn = []
        self.action_played = []

        self.starting_hands = []

        self.cards_left_when_lost = []
        self.opponent_cards_left_when_won = []

    def track_action(self, obs: BigTwoObservation, action: List[Card]):
        target = obs.last_cards_played
        current_hand = copy.deepcopy(obs.your_hands.cards)
        self.action_played.append(
            (obs.last_player_played, target, current_hand, action)
        )

    def track_game_end(self, env: BigTwo, starting_hands: List[List[Card]], done=False):
        ep_turns = env.get_hands_played()
        self.batch_hands_played.append(ep_turns)

        if not done:
            return

        self.batch_games_played += 1
        win = 1 if env.player_last_played == 0 else 0
        self.win_count.append(win)

        self.starting_hands.append(
            (starting_hands, env.player_last_played, env.get_starting_player_number())
        )

        if env.player_last_played != 0:
            self.cards_left_when_lost.append(len(env.player_hands[0]))
        else:
            opponent_hand_size = [
                len(hand)
                for player_number, hand in enumerate(env.player_hands)
                if player_number != 0
            ]
            self.opponent_cards_left_when_won += opponent_hand_size

        played_first_turn = 1 if env.get_starting_player_number() == 0 else 0
        self.played_first_turn.append(played_first_turn)

    def track_rewards(self, ep_ret, ep_len):
        self.batch_rets.append(ep_ret)
        self.batch_lens.append(ep_len)

    def summary(self) -> str:
        return (
            f"return: {np.mean(self.batch_rets):.3f}, "
            f"ep_len: {np.mean(self.batch_lens):.3f} "
            f"win rate: {self.win_rate():.3f} "
            f"ep_hands_played: {np.mean(self.batch_hands_played):.3f} "
            f"ep_games_played: {self.batch_games_played}, "
        )

    def __add__(self, other):
        result = SampleMetric()

        result.batch_lens = self.batch_lens + other.batch_lens
        result.batch_rets = self.batch_lens + other.batch_rets
        result.batch_hands_played = self.batch_hands_played + other.batch_hands_played
        result.batch_games_played = self.batch_games_played + other.batch_games_played

        result.win_count = self.win_count + other.win_count
        result.played_first_turn = self.played_first_turn + other.played_first_turn
        result.action_played = self.action_played + other.action_played

        result.cards_left_when_lost = (
            self.cards_left_when_lost + other.cards_left_when_lost
        )
        result.opponent_cards_left_when_won = (
            self.opponent_cards_left_when_won + other.opponent_cards_left_when_won
        )

        result.starting_hands = self.starting_hands + other.starting_hands

        return result

    def win_rate(self):
        return sum(self.win_count) / len(self.win_count)

    def win_rate_with_starting(self):
        starting_idx = np.where(np.array(self.played_first_turn) == 1)
        wrapped = np.array(self.win_count)
        return sum(wrapped[starting_idx]) / len(wrapped[starting_idx])

    def win_rate_without_starting(self):
        non_starting_idx = np.where(np.array(self.played_first_turn) == 0)
        wrapped = np.array(self.win_count)
        return sum(wrapped[non_starting_idx]) / len(wrapped[non_starting_idx])

    def start_game_rate(self):
        return sum(self.played_first_turn) / len(self.played_first_turn)


def get_current_dt_format() -> str:
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def sample_worker(input_queue: Queue, output: Queue):
    config_gpu()

    for bot_weight, opponent_weights, config, buffer_size in iter(
        input_queue.get, "STOP"
    ):
        buf, m = collect_data_from_env_self_play(
            bot_weight, opponent_weights, config, buffer_size
        )
        gc.collect()
        output.put((buf, m))


def profile_time(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()

        result = func(*args, **kwargs)

        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(20)
        print(s.getvalue())

        return result

    return wrapper


class SinglePlayerWrapper:
    def __init__(
        self, env: BigTwo, single_player_number: int, opponent_bots: List[Any]
    ):
        assert 0 <= single_player_number <= 3
        assert len(opponent_bots) == 3
        self.env = env
        self.player_number = single_player_number

        opponent_map = {}
        opponent_idx = 0
        for player_number in range(4):
            if player_number == single_player_number:
                continue

            opponent_map[player_number] = opponent_bots[opponent_idx]
            opponent_idx += 1

        self.opponent_bots = opponent_map

        self.starting_hands: List[List[Card]] = []

    def get_left_opponent_hand(self) -> BigTwoHand:
        left = 0 if (self.player_number + 1) > 3 else self.player_number + 1
        """
        0 -> 1
        1 -> 2
        2 -> 3
        3 -> 0
        """

        return self.env.player_hands[left]

    def get_top_opponent_hand(self) -> BigTwoHand:
        top = (
            self.player_number - 2
            if (self.player_number + 2) > 3
            else self.player_number + 2
        )
        """
        0 -> 2
        1 -> 3
        2 -> 0
        3 -> 1
        """

        return self.env.player_hands[top]

    def get_right_opponent_hand(self) -> BigTwoHand:
        right = (
            self.player_number - 1
            if (self.player_number + 3) > 3
            else self.player_number + 3
        )
        """
        0 -> 3
        1 -> 0
        2 -> 1
        3 -> 2
        """

        return self.env.player_hands[right]

    def step(self, action):
        obs, _, done = self.env.step(action.raw)
        if done:
            return obs, 1000, True

        # keep taking turns until it's targeted player turn again
        while True:
            current_obs = self.env.get_current_player_obs()
            if current_obs.current_player == self.player_number:
                return current_obs, 0, False

            opponent = self.opponent_bots[current_obs.current_player]
            action = opponent.action(current_obs)

            _, _, done = self.env.step(action.raw)
            if done:
                return self.env.get_player_obs(self.player_number), -1000, True

    def reset_and_start(self):
        obs = self.env.reset()
        self.starting_hands = [copy.deepcopy(x.cards) for x in self.env.player_hands]

        while True:
            if obs.current_player == self.player_number:
                return obs

            # if it's not the targeted player, we will keep playing until it is.
            opponent = self.opponent_bots[obs.current_player]
            action = opponent.action(obs)

            _, _, done = self.env.step(action.raw)

            # it should be never done since it is impossible to finish the game in the first round.
            assert not done

            # set the obs to the current player after previous player has played.
            obs = self.env.get_current_player_obs()


def build_opponent_bots(
    obs, bot_weights, opponent_weights: List[Any], config: ExperimentConfig
) -> List[Any]:
    policy_weight, value_weight = bot_weights
    opponent_bots = []
    for _ in range(3):
        if len(opponent_weights) > 0:
            opponent = BotBuilder.create_training_bot(config.bot_type, obs)
            pw, vw = random.choice(opponent_weights)
            opponent.set_weights(pw, vw)
        else:
            opponent = BotBuilder.create_training_bot(config.bot_type, obs)
            opponent.set_weights(policy_weight, value_weight)

        opponent_bots.append(opponent)

    return opponent_bots


def collect_data_from_env_self_play(
    bot_weights, opponent_weights: List[Any], config: ExperimentConfig, buffer_size=4000
) -> Tuple[PPOBufferInterface, SampleMetric]:
    buf = get_game_buf(config.bot_type)

    env = BigTwo()
    init_obs = env.reset()

    # player 0 is the one with the latest weight and the one we are training
    policy_weight, value_weight = bot_weights
    bot = BotBuilder.create_training_bot(config.bot_type, init_obs)
    bot.set_weights(policy_weight, value_weight)

    bot_ep_rews = []
    bot_buf = get_player_buf(config.bot_type)

    opponent_bots = build_opponent_bots(init_obs, bot_weights, opponent_weights, config)

    wrapped_env = SinglePlayerWrapper(
        env=env, opponent_bots=opponent_bots, single_player_number=0
    )

    obs = wrapped_env.reset_and_start()

    sample_metric = SampleMetric()
    timestep = 0
    while True:
        action = bot.action(obs)
        timestep += 1

        sample_metric.track_action(obs, action.cards)

        obs, reward, done = wrapped_env.step(action)

        bot_ep_rews.append(reward)
        bot_buf.store(
            obs=action.transformed_obs,
            act=action.cat,
            rew=reward,
            logp=action.logp,
            mask=action.mask,
        )

        epoch_ended = timestep == buffer_size
        if done or epoch_ended:
            ep_ret, ep_len = 0, 0

            ep_rews = bot_ep_rews
            ep_ret += sum(ep_rews)
            ep_len += len(ep_rews)

            estimated_values = bot.predict_value(bot_buf.get_curr_path())

            last_val = 0
            if not done and epoch_ended:
                transformed_obs = bot.transform_obs(obs)
                if isinstance(transformed_obs, Mapping):
                    transformed_obs = {
                        k: np.array([v]) for k, v in transformed_obs.items()
                    }
                else:
                    transformed_obs = np.array([transformed_obs])
                last_val = bot.predict_value(transformed_obs)

            bot_buf.finish_path(estimated_values.numpy().flatten(), last_val)
            buf.add(bot_buf)

            sample_metric.track_rewards(ep_ret, ep_len)
            sample_metric.track_game_end(
                wrapped_env.env, wrapped_env.starting_hands, done
            )

            # reset game
            obs = wrapped_env.reset_and_start()
            bot_buf = get_player_buf(config.bot_type)
            bot_ep_rews = []

        if epoch_ended:
            break

    return buf, sample_metric


def create_worker_task(
    num_cpu: int, bot_weight, opponent_weights: List[Any], config: ExperimentConfig
):
    chunk_buffer_size = config.buffer_size // num_cpu

    args = zip(
        itertools.repeat(bot_weight, num_cpu),
        itertools.repeat(opponent_weights, num_cpu),
        itertools.repeat(config, num_cpu),
        itertools.repeat(chunk_buffer_size, num_cpu),
    )

    return args


def display_top(snapshot, key_type="lineno", limit=10):
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print(
            "#%s: %s:%s: %.1f KiB"
            % (index, frame.filename, frame.lineno, stat.size / 1024)
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def display_memory_objects():
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)  # Prints out a summary of the large objects
    summary.print_(sum1)


def print_snapshot() -> tracemalloc.Snapshot:
    snapshot = tracemalloc.take_snapshot()
    # display_top(snapshot, limit=10)

    top_stats = snapshot.statistics("traceback")

    # pick the biggest memory block
    for i in range(10):
        # print("")
        stat = top_stats[i]
        # print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
        # for line in stat.traceback.format():
        #     print(line)
        # print("")

    return snapshot


class ExperimentLogger:
    def __init__(self, dir_path: str):
        self.dir_path = f"./experiments/{dir_path}"
        os.makedirs(self.dir_path)

        self.train_summary_writer = tf.summary.create_file_writer(
            f"{self.dir_path}/tensorboard"
        )

    def flush_config(self, config: ExperimentConfig):
        data = asdict(config)
        data["bot_type"] = config.bot_type.value
        # As we are using all scalar values, we need to pass an index
        # wrapping the dict in a list means the index of the values is 0
        config_df = pd.DataFrame([data])
        config_df.to_csv(f"{self.dir_path}/config.csv", index=False)

    def flush_metric(self, episode: int, metric: SampleMetric):
        self._flush_starting_hands(episode, metric)
        self._flush_action_history(episode, metric)
        self._flush_tensorboard(episode, metric)

    def _flush_action_history(self, episode: int, metric: SampleMetric):
        new_dir_path = f"./{self.dir_path}/action_history"
        os.makedirs(new_dir_path, exist_ok=True)

        if len(metric.action_played) > 0:
            pickle.dump(
                metric.action_played,
                open(f"{new_dir_path}/action_played_{episode}.pickle", "wb"),
            )

    def _flush_starting_hands(self, episode: int, metric: SampleMetric):
        new_dir_path = f"./{self.dir_path}/starting_hands"
        os.makedirs(new_dir_path, exist_ok=True)

        if len(metric.starting_hands) > 0:
            pickle.dump(
                metric.starting_hands,
                open(f"{new_dir_path}/starting_hands_{episode}.pickle", "wb"),
            )

    def _flush_tensorboard(self, episode: int, m: SampleMetric):
        with self.train_summary_writer.as_default():
            tf.summary.scalar("win_rate", m.win_rate(), step=episode)
            tf.summary.scalar("num_of_games", m.batch_games_played, step=episode)
            tf.summary.scalar("avg_rets", np.mean(m.batch_rets), step=episode)
            tf.summary.scalar(
                "system_memory_usage", psutil.virtual_memory().percent, step=episode
            )
            tf.summary.scalar(
                "win_rate_starting", m.win_rate_with_starting(), step=episode
            )
            tf.summary.scalar(
                "win_rate_without_starting", m.win_rate_without_starting(), step=episode
            )
            tf.summary.scalar(
                "avg_cards_left_when_lost",
                np.mean(m.cards_left_when_lost),
                step=episode,
            )
            tf.summary.scalar(
                "avg_cards_left_when_won",
                np.mean(m.opponent_cards_left_when_won),
                step=episode,
            )


def train():
    train_logger = get_logger("train_ppo", log_level=logging.INFO)
    env = BigTwo()

    # override config for serialise training
    config = ExperimentConfig()
    config.epoch = 10
    config.lr = 0.0001
    config.opponent_update_freq = 10
    config.buffer_size = 4000
    config.mini_batch_size = 1028
    config.bot_type = BotType.SIMPLE_PPO_BOT

    new_dir = f"{get_current_dt_format()}_test_run"
    experiment_logger = ExperimentLogger(new_dir)
    experiment_logger.flush_config(config)

    previous_bots = []
    bot = BotBuilder.create_training_bot(config.bot_type, env.reset(), lr=config.lr)
    for episode in range(config.epoch):
        sample_start_time = time.time()

        weights = bot.get_weights()

        buf, m = collect_data_from_env_self_play(
            weights, previous_bots, config, config.buffer_size
        )

        sample_time_taken = time.time() - sample_start_time

        if episode % config.opponent_update_freq == 0 and episode > 0:
            previous_bots.append(weights)

        if len(previous_bots) > config.opponent_buf_limit:
            previous_bots.pop(0)

        update_start_time = time.time()
        policy_loss, value_loss = bot.update(
            buf, mini_batch_size=config.mini_batch_size
        )
        update_time_taken = time.time() - update_start_time

        epoch_summary = (
            f"epoch: {episode + 1}, policy_loss: {policy_loss:.3f}, "
            f"value_loss: {value_loss:.3f}, "
            f"time to sample: {sample_time_taken} "
            f"time to update network: {update_time_taken}"
        )
        train_logger.info(epoch_summary)
        train_logger.info(m.summary())

        experiment_logger.flush_metric(episode, m)

        if episode % 2 == 0 and episode > 0:
            save_bot_dir_path = f"./experiments/{new_dir}/bot_save"
            os.makedirs(save_bot_dir_path, exist_ok=True)
            bot.agent.save(save_bot_dir_path)


def merge_result(
    results: List[Tuple[GameBuffer, SampleMetric]]
) -> Tuple[GameBuffer, SampleMetric]:
    result_buf, result_metric = None, None

    for buf, m in results:
        if result_buf is None:
            result_buf = buf
            result_metric = m
            continue

        result_buf += buf
        result_metric += m

    return result_buf, result_metric


def train_parallel(config: ExperimentConfig):
    mp.set_start_method("spawn")

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
    logging.getLogger("tensorflow").setLevel(logging.FATAL)

    # setting up worker for sampling
    NUMBER_OF_PROCESSES = config.num_of_worker
    task_queue = Queue()
    done_queue = Queue()

    # Start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        Process(target=sample_worker, args=(task_queue, done_queue)).start()

    previous_bots = []

    try:
        new_dir = get_current_dt_format()
        experiment_logger = ExperimentLogger(new_dir)
        experiment_logger.flush_config(config)

        train_logger = get_logger("train_ppo", log_level=logging.INFO)
        env = BigTwo()
        bot = BotBuilder.create_training_bot(config.bot_type, env.reset(), lr=config.lr)

        for episode in range(config.epoch):
            sample_start_time = time.time()
            gc.collect()

            weights = bot.get_weights()

            tasks = create_worker_task(
                NUMBER_OF_PROCESSES, weights, previous_bots, config
            )
            for task in tasks:
                task_queue.put(task)

            result: List[Tuple[GameBuffer, SampleMetric]] = []
            for i in range(NUMBER_OF_PROCESSES):
                result.append(done_queue.get())

            buf, m = merge_result(result)
            sample_time_taken = time.time() - sample_start_time

            if episode % config.opponent_update_freq == 0 and episode > 0:
                previous_bots.append(weights)

            if len(previous_bots) > config.opponent_buf_limit:
                previous_bots.pop(0)

            update_start_time = time.time()
            policy_loss, value_loss = bot.update(
                buf, mini_batch_size=config.mini_batch_size
            )
            update_time_taken = time.time() - update_start_time

            epoch_summary = (
                f"epoch: {episode + 1}, policy_loss: {policy_loss:.3f}, "
                f"value_loss: {value_loss:.3f}, "
                f"time to sample: {sample_time_taken:.3f} "
                f"time to update network: {update_time_taken:.3f}"
            )
            train_logger.info(epoch_summary)
            train_logger.info(m.summary())

            experiment_logger.flush_metric(episode, m)

            if episode % 100 == 0 or episode == config.epoch - 1:
                save_bot_dir_path = f"save/{new_dir}_{episode}"
                os.makedirs(save_bot_dir_path, exist_ok=True)
                bot.agent.save(save_bot_dir_path)

    finally:
        print("stopping workers")
        # Tell child processes to stop
        for i in range(NUMBER_OF_PROCESSES):
            task_queue.put("STOP")


if __name__ == "__main__":
    config_gpu()

    train()
    # train_parallel(ExperimentConfig())
