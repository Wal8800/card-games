import copy
import itertools
import pickle
import traceback

import tensorflow as tf

from collections import Counter
from multiprocessing import Process, Queue
from typing import Any, Callable, Dict, List

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from tqdm import trange

from bigtwo.bigtwo import BigTwoHand
from gamerunner.ppo_runner import build_bot
from playingcards.card import Card
from ray_runner.bigtwo_multi_agent import BigTwoMultiAgentEnv
from ray_runner.multi_agent_bigtwo_runner import ParametricActionsModel


def sample_worker(
    input_queue: Queue, output: Queue, build_bots: Callable[[], Dict[str, Any]]
):
    try:
        bots = build_bots()

        for hands in iter(input_queue.get, "STOP"):
            for player_order in itertools.permutations(bots.keys()):
                copies = copy.deepcopy(hands)
                bigtwo_hands = [BigTwoHand(hand) for hand in copies]
                benchmark_env = BigTwoMultiAgentEnv(bigtwo_hands)
                obs = benchmark_env.reset(with_raw_obs=True)

                # Mapping different bot to different player order eg
                # bot 1 = player 1, bot 2 = player 2, bot 3 = player 3, bot 4 = player 4
                # bot 1 = player 4, bot 2 = player 1, bot 3 = player 2, bot 4 = player 3
                player_id_to_bot_name = {
                    f"player_{i}": bot_name for i, bot_name in enumerate(player_order)
                }

                while True:
                    actions_mapping = {}
                    for player_id, individual_obs in obs.items():
                        bot_name = player_id_to_bot_name[player_id]
                        bot = bots[bot_name]
                        action_category = bot.action(individual_obs)
                        actions_mapping[player_id] = action_category

                    obs, rewards, done, _ = benchmark_env.step(
                        actions_mapping, with_raw_obs=True
                    )

                    if done["__all__"]:
                        for player_id, rew in rewards.items():
                            if rew > 0:
                                winning_bot_name = player_id_to_bot_name[player_id]
                                output.put(winning_bot_name)
                        break
    except Exception as e:
        traceback.print_exc()
        output.put("ERROR")


class RayBotMultiAgentWrapper:
    def __init__(self, ray_bot, ray_policy_key: str):
        self.ray_bot = ray_bot
        self.ray_policy_key = ray_policy_key

    def action(self, individual_obs):
        temp_obs = copy.deepcopy(individual_obs)
        del temp_obs["raw_obs"]
        return self.ray_bot.compute_single_action(
            temp_obs, policy_id=self.ray_policy_key
        )


class PPOBotMultiAgentWrapper:
    def __init__(self, ppo_bot):
        self.ppo_bot = ppo_bot

    def action(self, individual_obs):
        ppo_action = self.ppo_bot.action(individual_obs["raw_obs"])
        return ppo_action.cat


def create_bots():
    # Force it use CPU only by disabling GPU visible devices
    tf.config.set_visible_devices([], "GPU")

    env_name = "bigtwo_v1"
    register_env(env_name, lambda _: BigTwoMultiAgentEnv())

    ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)

    test_env = BigTwoMultiAgentEnv()
    default_policy_id = "policy_0"
    bigtwo_policies = {
        default_policy_id: (
            None,
            test_env.observation_space,
            test_env.action_space,
            {
                "model": {
                    "custom_model": "pa_model",
                },
                "gamma": 0.99,
            },
        )
    }

    config = (
        PPOConfig()
        .framework("tf2")
        .experimental(_disable_preprocessor_api=True)
        .multi_agent(
            policies=bigtwo_policies,
            policy_mapping_fn=lambda agent_id: default_policy_id,
        )
        .resources(
            num_gpus=0,
            num_cpus_per_worker=1,
            num_cpus_for_local_worker=1,
            num_gpus_per_worker=1,
        )
        .rollouts(num_rollout_workers=0)
    )

    ray_agent = config.build(env=env_name)
    ray_agent.restore(
        "/home/wal8800/workspace/card-games/ray_runner/"
        "results/bigtwo_v1/TracedPPO/TracedPPO_bigtwo_v1_67fef_00000_0_2022-09-28_21-49-04/checkpoint_010000/checkpoint-10000"
    )

    ppo_bot = build_bot(
        dir_path="../gamerunner/experiments/2021_07_13_21_31_32/bot_save/2021_07_13_21_31_32_19999",
    )

    return {
        "self_1": PPOBotMultiAgentWrapper(ppo_bot),
        "self_2": PPOBotMultiAgentWrapper(ppo_bot),
        "ray_1": RayBotMultiAgentWrapper(ray_agent, default_policy_id),
        "ray_2": RayBotMultiAgentWrapper(ray_agent, default_policy_id),
    }


def run_multi_agent_env_bench_mark():
    NUMBER_OF_PROCESSES = 10

    task_queue = Queue()
    output_queue = Queue()

    benchmark_result = []

    with open("./benchmark_hands_list_2021_08_07.pickle", "rb") as pickle_file:
        fixture: List[List[List[Card]]] = pickle.load(pickle_file)

    processes = []
    try:
        # Start worker processes
        for i in range(NUMBER_OF_PROCESSES):
            p = Process(
                target=sample_worker, args=(task_queue, output_queue, create_bots)
            )
            p.start()

            processes.append(p)

        for hands in fixture:
            task_queue.put(hands)

        task_queue.qsize()
        for _ in trange(24 * len(fixture)):
            result = output_queue.get()

            if result == "ERROR":
                break

            benchmark_result.append(result)
    finally:
        print("stopping workers")
        # Tell child processes to stop
        for i in range(NUMBER_OF_PROCESSES):
            task_queue.put("STOP")

        for p in processes:
            p.join(timeout=10)
            p.terminate()

    c = Counter(benchmark_result)
    print(c.most_common())


if __name__ == "__main__":
    # Run the benchmark CPU only because I can't run that many games at the same time with a single GPU.
    run_multi_agent_env_bench_mark()

    """
    [[('self', 15434), ('ray', 8566)]
    [('self', 14115), ('ray', 9885)]
    100%|██████████| 24000/24000 [07:20<00:00, 54.51it/s]
    [
        ('checkpoint-9766', 6597), 
        ('checkpoint-9766', 6550), 
        ('2021_07_13_21_31_32_19999', 5442), 
        ('2021_07_13_21_31_32_19999', 5411)
    ]
    """
