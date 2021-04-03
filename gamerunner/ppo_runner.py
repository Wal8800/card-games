import itertools
import logging
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from multiprocessing import Queue, Process
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from bigtwo.bigtwo import BigTwo
from gamerunner.cmd_line_bot import CommandLineBot
from gamerunner.ppo_bot import SimplePPOBot, PlayerBuffer, GameBuffer

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")


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


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_logger(logger_name, log_level):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.addHandler(get_console_handler())
    return logger


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

    def track_action(self, action: List[int], rewards):
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

    def summary(self) -> str:
        return (
            f"return: {np.mean(self.batch_rets):.3f}, "
            f"min returns: {np.min(self.batch_rets)}, "
            f"med returns: {np.median(self.batch_rets)}, "
            f"ep_len: {np.mean(self.batch_lens):.3f} "
            f"ep_hands_played: {np.mean(self.batch_hands_played):.3f} "
            f"ep_games_played: {self.batch_games_played}, "
            f"event counter: {self.events_counter}"
        )

    def cards_played_summary(self) -> str:
        return (
            f"num_of_cards_played_summary: {Counter(self.num_of_cards_played).most_common()}, "
            f"num_of_valid_cards_played_summary: {Counter(self.num_of_valid_card_played).most_common()}, "
        )


def create_player_buf(num_of_player=4) -> Dict[int, PlayerBuffer]:
    return {i: PlayerBuffer() for i in range(num_of_player)}


def create_player_rew_buf(num_of_player=4) -> Dict[int, List[int]]:
    return {i: [] for i in range(num_of_player)}


def sample_worker(input_queue: Queue, output: Queue):
    config_gpu()

    for policy_weight, value_weight, buffer_size in iter(input_queue.get, "STOP"):
        buf, m = collect_data_from_env(policy_weight, value_weight, buffer_size)
        output.put((buf, m))


def collect_data_from_env(
    policy_weight, value_weight, buffer_size=4000
) -> Tuple[GameBuffer, SampleMetric]:
    env = BigTwo()

    obs = env.reset()

    bot = SimplePPOBot(obs)
    bot.set_weights(policy_weight, value_weight)

    player_bufs = create_player_buf()
    player_ep_rews = create_player_rew_buf()

    sample_metric = SampleMetric()

    buf = GameBuffer()

    for t in range(buffer_size):
        obs = env.get_current_player_obs()
        action = bot.action(obs)

        new_obs, reward, done = env.step(action.raw)
        sample_metric.track_action(action.raw, reward)

        # storing the trajectory per players because the awards is per player not sum of all players.
        player_ep_rews[obs.current_player].append(reward)
        player_bufs[obs.current_player].store(
            action.obs_arr, action.cat, reward, action.logp, action.mask
        )

        epoch_ended = t == buffer_size - 1
        if done or epoch_ended:
            ep_ret, ep_len = 0, 0
            # add to buf
            # process each player buf then add it to the big one
            for player_number, player_buf in player_bufs.items():
                ep_rews = player_ep_rews[player_number]
                ep_ret += sum(ep_rews)
                ep_len += len(ep_rews)

                if player_buf.is_empty():
                    continue

                estimated_values = bot.predict_value(player_buf.get_curr_path())

                last_val = 0
                if not done and epoch_ended:
                    last_obs = env.get_player_obs(player_number)
                    last_obs_arr = np.array([bot.transform_obs(last_obs)])
                    last_val = bot.predict_value(last_obs_arr)

                player_buf.finish_path(estimated_values, last_val)
                buf.add(player_buf)

            sample_metric.track_rewards(ep_ret, ep_len)
            sample_metric.track_env(env, done)

            # reset game
            env.reset()
            player_bufs = create_player_buf()
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


def train(epoch=5, lr=0.001):
    train_logger = get_logger("train_ppo", log_level=logging.INFO)
    env = BigTwo()

    bot = SimplePPOBot(env.reset(), lr=lr)

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
            f"time to sample: {sample_time_taken} "
            f"time to update network: {update_time_taken}"
        )
        train_logger.info(epoch_summary)
        train_logger.info(m.summary())
        train_logger.info(m.cards_played_summary())


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


@dataclass
class ExperimentConfig:
    epoch: int = 5
    buffer_size: int = 4000
    lr: float = 0.0001
    mini_batch_size: int = 512

    num_of_worker: int = 10

    clip_ratio: float = 0.3


class ExperimentLogger:
    def __init__(self):
        self.metrics = []

    def store(self, m: SampleMetric):
        self.metrics.append(m)

    def flush(self, root_dir: str, config: ExperimentConfig):
        new_dir = datetime.now().strftime("%d_%b_%Y_%H_%M_%S")

        new_dir_path = f"./{root_dir}/{new_dir}"
        os.makedirs(new_dir_path)

        # As we are using all scalar values, we need to pass an index
        # wrapping the dict in a list means the index of the values is 0
        df = pd.DataFrame([asdict(config)])
        df.to_csv(f"{new_dir_path}/config.csv", index=False)

        min_ep_ret = [np.min(m.batch_rets) for m in self.metrics]
        avg_ep_ret = [np.mean(m.batch_rets) for m in self.metrics]
        med_ep_ret = [np.median(m.batch_rets) for m in self.metrics]
        plot_data = {
            "min_ret": min_ep_ret,
            "avg_ret": avg_ep_ret,
            "med_ret": med_ep_ret,
        }

        ax = sns.lineplot(data=plot_data)
        figure = ax.get_figure()
        figure.savefig(f"{new_dir_path}/returns_plot.png", dpi=400)


def train_parallel(config: ExperimentConfig):
    mp.set_start_method("spawn")

    # setting up worker for sampling
    NUMBER_OF_PROCESSES = config.num_of_worker
    task_queue = Queue()
    done_queue = Queue()

    # Start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        Process(target=sample_worker, args=(task_queue, done_queue)).start()

    try:
        train_logger = get_logger("train_ppo", log_level=logging.INFO)
        env = BigTwo()
        bot = SimplePPOBot(env.reset(), lr=config.lr)
        experiment_log = ExperimentLogger()

        ep_returns, min_ep_returns, med_ep_ret = [], [], []
        for i_episode in range(config.epoch):
            sample_start_time = time.time()

            policy_weight, value_weight = bot.get_weights()

            tasks = create_worker_task(
                NUMBER_OF_PROCESSES, policy_weight, value_weight, config.buffer_size
            )
            for task in tasks:
                task_queue.put(task)

            result: List[Tuple[GameBuffer, SampleMetric]] = []
            for i in range(NUMBER_OF_PROCESSES):
                result.append(done_queue.get())

            buf, m = merge_result(result)
            sample_time_taken = time.time() - sample_start_time

            update_start_time = time.time()
            policy_loss, value_loss = bot.update(
                buf, mini_batch_size=config.mini_batch_size
            )
            update_time_taken = time.time() - update_start_time

            epoch_summary = (
                f"epoch: {i_episode + 1}, policy_loss: {policy_loss:.3f}, "
                f"value_loss: {value_loss:.3f}, "
                f"time to sample: {sample_time_taken:.3f} "
                f"time to update network: {update_time_taken:.3f}"
            )
            train_logger.info(epoch_summary)
            train_logger.info(m.summary())
            train_logger.info(m.cards_played_summary())

            experiment_log.store(m)

        experiment_log.flush("./experiments", config)
        # bot.save("save")
    finally:
        print("stopping workers")
        # Tell child processes to stop
        for i in range(NUMBER_OF_PROCESSES):
            task_queue.put("STOP")


def play_with_cmd():
    env = BigTwo()
    init_obs = env.reset()

    player_list = []
    for i in range(BigTwo.number_of_players() - 1):
        bot = SimplePPOBot(init_obs)
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
            action = current_player.action(obs)
            new_obs, reward, done = env.step(action.raw)

        episode_step += 1
        print("action: " + str(action))
        print(f"after player hand: {new_obs.your_hands}")
        print(env.event_count)
        print("====")

    env.display_all_player_hands()


if __name__ == "__main__":
    config_gpu()
    start_time = time.time()
    train()
    # train_parallel(ExperimentConfig())
    # play_with_cmd()
    print(f"Time taken: {time.time() - start_time:.3f} seconds")
