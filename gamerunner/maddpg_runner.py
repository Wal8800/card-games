from bigtwo import BigTwo, BigTwoRealTime
import time
import tensorflow as tf
import argparse
import numpy as np
import pickle

import maddpg.maddpg.common.tf_util as U
from maddpg.maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
    return out


def get_trainers(env, num_players, obs_shape, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_players):
        trainers.append(
            trainer(
                "agent_%d" % i,
                model,
                obs_shape,
                [env.action_space for _ in range(4)],
                i,
                arglist
            )
        )
    return trainers


def flatten_obs(obs):
    flat_obs = []
    for o in obs:
        if isinstance(o, (list, np.ndarray)):
            flat_obs.extend(o)
        else:
            flat_obs.append(o)
    return np.array(flat_obs)


def train(arglist):
    with U.single_threaded_session():
        env = BigTwoRealTime()

        space_list = []
        for sub_space in env.observation_space.spaces:
            space_list.append(sub_space.shape if len(sub_space.shape) > 0 else (1,))

        final_space = (np.sum(space_list),)
        obs_shape_n = [final_space for _ in range(4)]
        trainers = get_trainers(env, BigTwo.number_of_players(), obs_shape_n, arglist)

        U.initialize()

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        while True:
            flat_obs_n = [flatten_obs(obs) for obs in obs_n]
            action_n = [agent.action(obs) for agent, obs in zip(trainers, flat_obs_n)]
            new_obs_n, rew_n, done_n = env.step(action_n)
            new_flat_obs_n = [flatten_obs(obs) for obs in new_obs_n]

            # print("turn ", episode_step)
            # print("current_player", obs[3])
            # print('before player hand:' + Card.display_cards_string(obs[2]))
            # print('last_player_played: ', obs[4])
            # print('cards played: ' + Card.display_cards_string(obs[1]))
            # print('action: ' + str(action[1]))
            # print('after player hand:' + Card.display_cards_string(new_obs[2]))
            # print("====")

            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)

            for i, agent in enumerate(trainers):
                agent.experience(flat_obs_n[i], action_n[i], rew_n[i], new_flat_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                # env.display_all_player_hands()
                env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # for benchmarking learned policies
            # if arglist.benchmark:
            #     for i, info in enumerate(info_n):
            #         agent_info[-1][i].append(info_n['n'])
            #     if train_step > arglist.benchmark_iters and (done or terminal):
            #         file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
            #         print('Finished benchmarking, now saving...')
            #         with open(file_name, 'wb') as fp:
            #             pickle.dump(agent_info[:-1], fp)
            #         break
            #     continue

            # increment global step counter
            train_step += 1

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                    round(time.time() - t_start, 3)))
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
