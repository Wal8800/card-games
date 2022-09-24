import copy
import itertools
import multiprocessing as mp
import pickle
import traceback
from collections import Counter
from multiprocessing import Process, Queue
from typing import Any, Callable, Dict, List

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from tqdm import tqdm, trange

from bigtwo.bigtwo import BigTwoHand
from gamerunner.ppo_runner import build_bot
from playingcards.card import Card
from ray_runner.bigtwo_multi_agent import BigTwoMultiAgentEnv
from ray_runner.multi_agent_bigtwo_runner import (
    CustomMetricCallback,
    ParametricActionsModel,
)


def serial_run():
    ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
    env_name = "bigtwo_v1"
    register_env(env_name, lambda _: BigTwoMultiAgentEnv())

    test_env = BigTwoMultiAgentEnv()
    gen_obs_space = test_env.observation_space

    gen_act_space = test_env.action_space

    def gen_policy(i):
        config = {
            "model": {
                "custom_model": "pa_model",
            },
            "gamma": 0.99,
        }
        return None, gen_obs_space, gen_act_space, config

    bigtwo_policies = {"policy_0": gen_policy(0)}

    policy_ids = list(bigtwo_policies.keys())

    run_config = {
        # Environment specific
        "env": env_name,
        # General
        "callbacks": CustomMetricCallback,
        "_disable_preprocessor_api": True,
        "framework": "tf2",
        "num_gpus": 1,
        "num_workers": 0,
        # Method specific
        "multiagent": {
            "policies": bigtwo_policies,
            "policy_mapping_fn": (lambda agent_id: policy_ids[0]),
        },
    }

    agent = PPOTrainer(run_config)
    agent.restore(
        "/home/wal8800/workspace/card-games/ray_runner/results/bigtwo_v1/PPO/CustomPPO_bigtwo_v1_acf0b_00000_0_2022-09-12_07-19-37/checkpoint_009766/checkpoint-9766"
    )

    policy = agent.get_policy(policy_id="policy_0")
    print(policy.model.action_model.base_model.summary())

    # agent_two = PPOTrainer(run_config)
    # agent_two.restore(
    #     "../ray_runner/ray_results/bigtwo_v1/PPO/PPO_bigtwo_v1_54920_00000_0_2022-07-11_08-24-42/checkpoint_001226/checkpoint-1226"
    # )

    og_ppo_bot = build_bot(
        dir_path="../gamerunner/experiments/2021_07_13_21_31_32/bot_save/2021_07_13_21_31_32_10000",
    )

    print(og_ppo_bot.agent.policy.summary())

    # sys.exit(0)

    obs = test_env.reset(with_raw_obs=True)

    with open("./benchmark_hands_list_2021_08_07.pickle", "rb") as pickle_file:
        fixtures: List[List[List[Card]]] = pickle.load(pickle_file)

    bot_mapping = {0: "ray", 1: "ray", 2: "self", 3: "self"}

    player_id_idx_mapping = {f"player_{i}": i for i in range(4)}
    benchmark_result = []

    player_order_perm = itertools.permutations(bot_mapping.keys())

    for fixture, player_order in tqdm(
        itertools.product(fixtures, player_order_perm), total=24 * len(fixtures)
    ):
        copies = copy.deepcopy(fixture)
        bigtwo_hands = [BigTwoHand(hand) for hand in copies]

        benchmark_env = BigTwoMultiAgentEnv(bigtwo_hands)
        obs = benchmark_env.reset(with_raw_obs=True)

        player_id_bot_type = {
            f"player_{i}": bot_mapping[player_order[i]] for i in range(4)
        }
        while True:
            actions_mapping = {}
            for player_id, individual_obs in obs.items():
                bot_type = player_id_bot_type[player_id]
                if bot_type == "ray":
                    temp_obs = copy.deepcopy(individual_obs)
                    del temp_obs["raw_obs"]
                    action = agent.compute_single_action(temp_obs, policy_id="policy_0")
                    actions_mapping[player_id] = action
                else:
                    ppo_action = og_ppo_bot.action(individual_obs["raw_obs"])
                    actions_mapping[player_id] = ppo_action.cat

            obs, rewards, done, _ = benchmark_env.step(
                actions_mapping, with_raw_obs=True
            )

            if done["__all__"]:
                for player_id, rew in rewards.items():
                    if rew > 0:
                        winning_bot_type = player_id_bot_type[player_id]
                        benchmark_result.append(winning_bot_type)
                break

    c = Counter(benchmark_result)
    print(c.most_common())


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
    env_name = "bigtwo_v1"
    register_env(env_name, lambda _: BigTwoMultiAgentEnv())

    ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)

    test_env = BigTwoMultiAgentEnv()
    gen_obs_space = test_env.observation_space
    gen_act_space = test_env.action_space

    ray_policy_key = "policy_0"

    def gen_policy(i):
        config = {
            "model": {
                "custom_model": "pa_model",
            },
            "gamma": 0.99,
        }
        return None, gen_obs_space, gen_act_space, config

    bigtwo_policies = {ray_policy_key: gen_policy(0)}
    run_config = {
        # Environment specific
        "env": env_name,
        # General
        "_disable_preprocessor_api": True,
        "framework": "tf2",
        "num_gpus": 0,
        "num_workers": 0,
        "eager_tracing": True,
        # Method specific
        "multiagent": {
            "policies": bigtwo_policies,
            "policy_mapping_fn": (lambda agent_id: ray_policy_key),
        },
    }

    ray_agent = PPOTrainer(config=run_config, env=env_name)
    ray_agent.restore(
        "/home/wal8800/workspace/card-games/ray_runner/results/bigtwo_v1/PPO/"
        "CustomPPO_bigtwo_v1_acf0b_00000_0_2022-09-12_07-19-37/checkpoint_009766/checkpoint-9766"
    )
    # policy = ray_agent.get_policy(policy_id="policy_0")

    ppo_bot = build_bot(
        dir_path="../gamerunner/experiments/2021_07_13_21_31_32/bot_save/2021_07_13_21_31_32_19999",
    )

    return {
        "self_1": PPOBotMultiAgentWrapper(ppo_bot),
        "self_2": PPOBotMultiAgentWrapper(ppo_bot),
        "ray_1": RayBotMultiAgentWrapper(ray_agent, ray_policy_key),
        "ray_2": RayBotMultiAgentWrapper(ray_agent, ray_policy_key),
    }


def run_multi_agent_env_bench_mark():
    NUMBER_OF_PROCESSES = 12

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
    # TODO(wal8800) Need to upgrade ray to the latest, otherwise ray 1.8 there is a bug in eager tf policy not respecting num_gpus
    # Needed to set CUDA_VISIBLE_DEVICES = -1 to disable GPU as ray 1.8 not respecting num_gpus
    # serial_run()
    run_multi_agent_env_bench_mark()
    # [('self', 15434), ('ray', 8566)]
    # [('self', 14115), ('ray', 9885)]

    """
    100%|██████████| 24000/24000 [07:20<00:00, 54.51it/s]
    [
        ('checkpoint-9766', 6597), 
        ('checkpoint-9766', 6550), 
        ('2021_07_13_21_31_32_19999', 5442), 
        ('2021_07_13_21_31_32_19999', 5411)
    ]
    """
