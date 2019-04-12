from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2, A2C
from bigtwo.valid_cards_game import ValidCardGame, pick_cards_from_hand, ValidCardGameAlt, from_one_hot_encoding
from playingcards.card import Card
import numpy as np
import tensorflow as tf


def evaluate(model, env):
    obs = env.reset()
    total_rewards = 0
    for i in range(20):
        action, _states = model.predict(obs)
        new_obs, rewards, dones, info = env.step(action)

        if rewards[0] > 0:
            hand = [Card.from_number(x) for x in obs[0]]
            # obs_list = obs[0]
            # hand = []
            # for j in range(0, 26, 2):
            #     hand.append((obs_list[j] * 13) + obs_list[j+1])
            #
            # hand = [Card.from_number(x) for x in hand]

            cards = pick_cards_from_hand(action[0], hand)
            hand_str = Card.display_cards_string(obs[0])
            # hand_str = ' '.join(str(x) for x in hand)
            card_str = ' '.join(str(x) for x in cards)
            print("hand: {:s} selected: {:s} reward: {:f}".format(hand_str, card_str, rewards[0]))

        obs = new_obs
        total_rewards += rewards[0]
        # env.render()

    print("total rewards: ", total_rewards)


def evaluate_alt(model, env):
    obs = env.reset()
    total_rewards = 0
    for i in range(20):
        action, _states = model.predict(obs)
        new_obs, rewards, dones, info = env.step(action)

        if rewards > 0:
            hand = from_one_hot_encoding(obs[0])
            cards = pick_cards_from_hand(action[0], hand)
            hand_str = ' '.join(str(x) for x in hand)
            card_str = ' '.join(str(x) for x in cards)
            print("hand: {:s} selected: {:s} reward: {:f}".format(hand_str, card_str, rewards[0]))

        obs = new_obs
        total_rewards += rewards[0]
        # env.render()

    print("total rewards: ", total_rewards)


def evaluate_parallel(model, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward
    """
    episode_rewards = [[0.0] for _ in range(env.num_envs)]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        actions, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(actions)

        # Stats
        for i in range(env.num_envs):
            episode_rewards[i][-1] += rewards[i]
            if dones[i]:
                episode_rewards[i].append(0.0)

    mean_rewards = [0.0 for _ in range(env.num_envs)]
    n_episodes = 0
    for i in range(env.num_envs):
        mean_rewards[i] = np.mean(episode_rewards[i])
        n_episodes += len(episode_rewards[i])

        # Compute mean reward
    mean_reward = round(np.mean(mean_rewards), 1)
    print("Mean reward:", mean_reward, "Num episodes:", n_episodes)

    return mean_reward


def base_env_func():
    return ValidCardGameAlt()


# base_env = ValidCardGame()
base_env = ValidCardGameAlt()
num_cpu = 8
env = DummyVecEnv([lambda: base_env])
# env = SubprocVecEnv([base_env_func for i in range(num_cpu)])
agent = PPO2(MlpPolicy, env, verbose=1, learning_rate=0.01)
agent.learn(total_timesteps=1000000)


# evaluate(model, env)
evaluate_alt(agent, env)
# evaluate_parallel(agent)




