import cProfile
import gc
import io
import itertools
import linecache
import logging
import multiprocessing as mp
import os
import pstats
import random
import sys
import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from multiprocessing import Queue, Process
from pstats import SortKey
from typing import Dict, List, Tuple, Mapping, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from algorithm.agent import PPOBufferInterface
from pympler import summary, muppy

from bigtwo.bigtwo import BigTwo
from gamerunner.ppo_bot import (
    SimplePPOBot,
    GameBuffer,
    PlayerBuffer,
    RandomPPOBot,
)

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


@dataclass
class ExperimentConfig:
    epoch: int = 5000
    lr: float = 0.0001
    buffer_size: int = 4000
    mini_batch_size: int = 512

    num_of_worker: int = 8

    clip_ratio: float = 0.3

    opponent_buf_limit = 10
    opponent_update_freq = 100

    bot_class = SimplePPOBot
    game_buf_class = GameBuffer
    player_buf_class = PlayerBuffer

    # bot_class = EmbeddedInputBot
    # game_buf_class = MultiInputGameBuffer
    # player_buf_class = MultiInputPlayerBuffer


class SampleMetric:
    def __init__(self):
        self.batch_lens = []
        self.batch_rets = []
        self.batch_hands_played = []
        self.num_of_cards_played = []
        self.num_of_valid_card_played = []
        self.game_history = []
        self.action_history = []
        self.batch_games_played: int = 0
        self.win_count = {}
        self.loss_count = {}

    def track_action(self, action: List[int], rewards):
        num_of_cards = sum(action)

        self.num_of_cards_played.append(num_of_cards)

    def track_env(self, env: BigTwo, done=False):
        ep_turns = env.get_hands_played()
        self.batch_hands_played.append(ep_turns)

        if done:
            self.batch_games_played += 1

            self.win_count[env.player_last_played] = (
                self.win_count.get(env.player_last_played, 0) + 1
            )

            for i in range(env.number_of_players()):
                if i == env.player_last_played:
                    continue

                self.loss_count[i] = self.loss_count.get(i, 0) + 1

        self.game_history.append(env.state)
        self.action_history.append(env.past_actions)

    def track_rewards(self, ep_ret, ep_len):
        self.batch_rets.append(ep_ret)
        self.batch_lens.append(ep_len)

    def summary(self) -> str:
        return (
            f"return: {np.mean(self.batch_rets):.3f}, "
            f"ep_len: {np.mean(self.batch_lens):.3f} "
            f"win rate: {self.win_rate(0):.3f} "
            f"ep_hands_played: {np.mean(self.batch_hands_played):.3f} "
            f"ep_games_played: {self.batch_games_played}, "
        )

    def cards_played_summary(self) -> str:
        return f"num_of_cards_played_summary: {Counter(self.num_of_cards_played).most_common()}, "

    def __add__(self, other):
        result = SampleMetric()

        result.batch_lens = self.batch_lens + other.batch_lens
        result.batch_rets = self.batch_lens + other.batch_rets
        result.batch_hands_played = self.batch_hands_played + other.batch_hands_played
        result.num_of_cards_played = (
            self.num_of_cards_played + other.num_of_cards_played
        )
        result.game_history = self.game_history + [
            h for h in other.game_history if len(h) > 0
        ]
        result.action_history = self.action_history + [
            h for h in other.action_history if len(h) > 0
        ]
        result.batch_games_played = self.batch_games_played + other.batch_games_played
        result.num_of_valid_card_played = (
            self.num_of_valid_card_played + other.num_of_valid_card_played
        )

        result.win_count = {
            i: self.win_count.get(i, 0) + other.win_count.get(i, 0)
            for i in range(BigTwo.number_of_players())
        }

        result.loss_count = {
            i: self.loss_count.get(i, 0) + other.loss_count.get(i, 0)
            for i in range(BigTwo.number_of_players())
        }

        return result

    def win_rate(self, player_number: int):
        return self.win_count.get(player_number, 0) / (
            self.win_count.get(player_number, 0) + self.loss_count.get(player_number, 0)
        )


def get_current_dt_format() -> str:
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def flush_config(root_dir: str, new_dir: str, config: ExperimentConfig):
    new_dir_path = f"./{root_dir}/{new_dir}"
    os.makedirs(new_dir_path)

    data = asdict(config)
    data["bot_class_name"] = config.bot_class.__name__
    # As we are using all scalar values, we need to pass an index
    # wrapping the dict in a list means the index of the values is 0
    config_df = pd.DataFrame([data])
    config_df.to_csv(f"{new_dir_path}/config.csv", index=False)


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


def collect_data_from_env_random_bot(
    bot_weights, opponent_weights: List[Any], config: ExperimentConfig, buffer_size=4000
) -> Tuple[PPOBufferInterface, SampleMetric]:
    env = BigTwo()
    obs = env.reset()

    policy_weight, value_weight = bot_weights

    bot = config.bot_class(obs)
    bot.set_weights(policy_weight, value_weight)

    buf = config.game_buf_class()

    bot_ep_rews = []
    bot_buf = config.player_buf_class()

    sample_metric = SampleMetric()
    random_bot = RandomPPOBot()
    timestep = 0
    while True:
        obs = env.get_current_player_obs()

        if obs.current_player == 0:
            action = bot.action(obs)
            timestep += 1
        else:
            action = random_bot.action(obs)

        new_obs, reward, done = env.step(action.raw)
        sample_metric.track_action(action.raw, reward)

        if obs.current_player == 0:
            # storing the trajectory per players because the awards is per player not sum of all players.
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
                last_obs = env.get_player_obs(0)
                transformed_obs = bot.transform_obs(last_obs)
                if not isinstance(transformed_obs, Mapping):
                    transformed_obs = np.array([transformed_obs])
                last_val = bot.predict_value(transformed_obs)
            bot_buf.finish_path(estimated_values.numpy().flatten(), last_val)
            buf.add(bot_buf)

            sample_metric.track_rewards(ep_ret, ep_len)
            sample_metric.track_env(env, done)

            # reset game
            env.reset()
            bot_buf = config.player_buf_class()
            bot_ep_rews = []

        if epoch_ended:
            break

    return buf, sample_metric


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
            opponent = config.bot_class(obs)
            pw, vw = random.choice(opponent_weights)
            opponent.set_weights(pw, vw)
        else:
            opponent = config.bot_class(obs)
            opponent.set_weights(policy_weight, value_weight)

        opponent_bots.append(opponent)

    return opponent_bots


def collect_data_from_env_self_play(
    bot_weights, opponent_weights: List[Any], config: ExperimentConfig, buffer_size=4000
) -> Tuple[PPOBufferInterface, SampleMetric]:
    buf = config.game_buf_class()

    env = BigTwo()
    init_obs = env.reset()

    # player 0 is the one with the latest weight and the one we are training
    policy_weight, value_weight = bot_weights
    bot = config.bot_class(init_obs)
    bot.set_weights(policy_weight, value_weight)

    bot_ep_rews = []
    bot_buf = config.player_buf_class()

    opponent_bots = build_opponent_bots(init_obs, bot_weights, opponent_weights, config)

    wrapped_env = SinglePlayerWrapper(
        env=env, opponent_bots=opponent_bots, player_number=0
    )

    obs = wrapped_env.reset_and_start()

    sample_metric = SampleMetric()
    timestep = 0
    while True:
        action = bot.action(obs)
        timestep += 1

        obs, reward, done = wrapped_env.step(action)
        sample_metric.track_action(action.raw, reward)

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
                if not isinstance(transformed_obs, Mapping):
                    transformed_obs = np.array([transformed_obs])
                last_val = bot.predict_value(transformed_obs)

            bot_buf.finish_path(estimated_values.numpy().flatten(), last_val)
            buf.add(bot_buf)

            sample_metric.track_rewards(ep_ret, ep_len)
            sample_metric.track_env(wrapped_env.env, done)

            # reset game
            obs = wrapped_env.reset_and_start()
            bot_buf = config.player_buf_class()
            bot_ep_rews = []

        if epoch_ended:
            break

    return buf, sample_metric


def collect_data_from_env(
    bot_weights, opponent_weights: List[Any], config: ExperimentConfig, buffer_size=4000
) -> Tuple[PPOBufferInterface, SampleMetric]:
    env = BigTwo()

    obs = env.reset()

    policy_weight, value_weight = bot_weights

    bot = config.bot_class(obs)
    bot.set_weights(policy_weight, value_weight)

    # player 0 is the one with the latest weight and the one we are training
    players = {0: bot}

    # player 1 to 3 are opponents with previous weights if available.
    for i in range(3):
        opponent = config.bot_class(obs)

        # used latest weights if we haven't sampled any previous opponent weights
        if len(opponent_weights) == 0:
            opponent.set_weights(policy_weight, value_weight)
        else:
            oppo_p_weights, oppo_v_weights = random.choice(opponent_weights)
            opponent.set_weights(oppo_p_weights, oppo_v_weights)

        players[i + 1] = opponent

    player_bufs = create_player_buf(config.player_buf_class)
    buf = config.game_buf_class()

    player_ep_rews = create_player_rew_buf()

    sample_metric = SampleMetric()
    for t in range(buffer_size):
        obs = env.get_current_player_obs()

        current_player = players[obs.current_player]

        action = current_player.action(obs)

        new_obs, reward, done = env.step(action.raw)
        sample_metric.track_action(action.raw, reward)

        # storing the trajectory per players because the awards is per player not sum of all players.
        player_ep_rews[obs.current_player].append(reward)
        player_bufs[obs.current_player].store(
            action.transformed_obs, action.cat, reward, action.logp, action.mask
        )

        epoch_ended = t == buffer_size - 1
        if done or epoch_ended:
            ep_ret, ep_len = 0, 0
            # add to buf
            # process each player buf then add it to the big one
            for player_number, player_buf in player_bufs.items():
                if player_buf.is_empty():
                    continue

                ep_rews = player_ep_rews[player_number]
                ep_ret += sum(ep_rews)
                ep_len += len(ep_rews)

                player = players[player_number]

                estimated_values = player.predict_value(player_buf.get_curr_path())

                last_val = 0
                if not done and epoch_ended:
                    last_obs = env.get_player_obs(player_number)
                    transformed_obs = player.transform_obs(last_obs)
                    if not isinstance(transformed_obs, Mapping):
                        transformed_obs = np.array([transformed_obs])
                    last_val = player.predict_value(transformed_obs)
                player_buf.finish_path(estimated_values.numpy().flatten(), last_val)
                buf.add(player_buf)

            sample_metric.track_rewards(ep_ret, ep_len)
            sample_metric.track_env(env, done)

            # reset game
            env.reset()
            player_bufs = create_player_buf(config.player_buf_class)
            player_ep_rews = create_player_rew_buf()

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


def print_snapshot():
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot, limit=20)

    top_stats = snapshot.statistics("traceback")

    # pick the biggest memory block
    for i in range(5):
        print("")
        stat = top_stats[i]
        print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
        for line in stat.traceback.format():
            print(line)
        print("")


def train():
    train_logger = get_logger("train_ppo", log_level=logging.INFO)
    env = BigTwo()

    # override config for serialise training
    config = ExperimentConfig()
    config.epoch = 10
    config.lr = 0.0001
    config.opponent_update_freq = 25
    config.buffer_size = 1000
    config.mini_batch_size = 128

    new_dir = get_current_dt_format()
    # flush_config("experiments", new_dir, config)
    # train_summary_writer = tf.summary.create_file_writer(f"tensorboard/{new_dir}")

    previous_bots = []
    bot = config.bot_class(env.reset(), lr=config.lr)
    for episode in range(config.epoch):

        print("episode: ", episode, "len: ", len(previous_bots))

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
        train_logger.info(m.cards_played_summary())

        # with train_summary_writer.as_default():
        #     tf.summary.scalar("win_rate", m.win_rate(0), step=episode)
        #     tf.summary.scalar("num_of_games", m.batch_games_played, step=episode)
        #     tf.summary.scalar("avg_rets", np.mean(m.batch_rets), step=episode)


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
        flush_config("experiments", new_dir, config)
        train_summary_writer = tf.summary.create_file_writer(f"tensorboard/{new_dir}")
        train_logger = get_logger("train_ppo", log_level=logging.INFO)
        env = BigTwo()
        bot = config.bot_class(env.reset(), lr=config.lr)

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
            train_logger.info(m.cards_played_summary())

            with train_summary_writer.as_default():
                tf.summary.scalar("win_rate", m.win_rate(0), step=episode)
                tf.summary.scalar("num_of_games", m.batch_games_played, step=episode)
                tf.summary.scalar("avg_rets", np.mean(m.batch_rets), step=episode)

        # bot.save("save")
    finally:
        print("stopping workers")
        # Tell child processes to stop
        for i in range(NUMBER_OF_PROCESSES):
            task_queue.put("STOP")


if __name__ == "__main__":
    config_gpu()
    start_time = time.time()
    # train()
    train_parallel(ExperimentConfig())
    # play_with_cmd()
    print(f"Time taken: {time.time() - start_time:.3f} seconds")
