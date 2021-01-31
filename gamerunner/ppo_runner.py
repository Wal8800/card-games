import numpy as np
import scipy.signal
import scipy.signal
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers, models
from tensorflow_probability import distributions as tfd

from bigtwo import BigTwo
from playingcards.card import Card
from random_bot import RandomBot


def discounted_sum_of_rewards(rewards, gamma):
    """
    [1 2 3]

    [
      0.99^0 * 1 + 0.99^1 * 2 + 0.99^2 * 3
      0.99^0 * 2 + 0.99^1 * 3
      0.99^0 * 3
    ]
    """
    return scipy.signal.lfilter([1], [1, float(-gamma)], rewards[::-1], axis=0)[::-1]


class PPOBuffer:
    def __init__(self, obs_shape, size: int, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, *obs_shape), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, logp):
        assert self.ptr < self.max_size

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp

        self.ptr += 1

    def get_curr_path(self) -> np.ndarray:
        path_slice = slice(self.path_start_idx, self.ptr)
        return self.obs_buf[path_slice]

    def finish_path(self, estimated_vals: np.array, last_val=0.0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        # δVt = rt + γV(st+1) − V(st)
        vals = np.append(estimated_vals, last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discounted_sum_of_rewards(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discounted_sum_of_rewards(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        return self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf


class BigTwoPPOAgent:
    def __init__(self, obs_shape, act_dim, env_name: str, dir_path: str = None):
        self.obs_shape = obs_shape
        self.act_dim = act_dim
        self.env_name = env_name
        self.train_v_iters = 80
        self.train_p_iters = 80
        self.target_kl = 0.01
        self.clip_ratio = 0.3

        # create policy and value function from saved files.
        if dir_path is not None:
            custom_obj = {"PPO": PPO}
            self.policy = models.load_model(f"{dir_path}/{self.env_name}_policy.h5", custom_objects=custom_obj)
            self.value_function = models.load_model(f"{dir_path}/{self.env_name}_value_function.h5",
                                                    custom_objects=custom_obj)
            return

        # creating the policy
        obs_inp = layers.Input(shape=obs_shape, name="obs")
        x = layers.Dense(64, activation="relu")(obs_inp)
        x = layers.Dense(64, activation="relu")(x)
        output = layers.Dense(act_dim)(x)
        self.policy = PPO(clip_ratio=self.clip_ratio, inputs=obs_inp, outputs=output)
        self.policy.compile(optimizer=optimizers.Adam(learning_rate=0.01))

        value_input = layers.Input(shape=obs_shape, name="obs")
        x = layers.Dense(64, activation="relu")(value_input)
        x = layers.Dense(64, activation="relu")(x)
        value_output = layers.Dense(1)(x)
        self.value_function = keras.Model(inputs=value_input, outputs=value_output)
        self.value_function.compile(optimizer=optimizers.Adam(0.01), loss=keras.losses.MeanSquaredError())

    def action(self, observation):
        wrapped = np.array([observation])
        dist = tfd.Bernoulli(logits=self.policy(wrapped))
        sampled_action = dist.sample().numpy()
        return sampled_action[0], dist.log_prob(sampled_action).numpy()[0]

    def update(self, buf: PPOBuffer):
        obs_buf, act_buf, adv_buf, ret_buf, logp_buf = buf.get()
        result, value_loss = {}, {}
        # update the value function
        for _ in range(self.train_v_iters):
            value_loss = self.value_function.train_step((obs_buf, tf.constant(ret_buf)))

        # update the policy by a single step
        for i in range(self.train_p_iters):
            result = self.policy.train_step([obs_buf, act_buf, adv_buf, logp_buf])
            kl = result["approx_kl"]
            if kl > 1.5 * self.target_kl:
                print(f"Early stopping at step {i} due to reaching max kl: {kl}")
                break

        return result["loss"].numpy(), value_loss['loss'].numpy()

    def predict_value(self, observations):
        return self.value_function.predict(observations)

    def save(self, dir_path: str):
        self.policy.save(f"{dir_path}/{self.env_name}_policy.h5")
        self.value_function.save(f"{dir_path}/{self.env_name}_value_function.h5")


class PPO(keras.Model):
    def __init__(self, clip_ratio=0.2, **kwargs):
        self.clip_ratio = clip_ratio
        super(PPO, self).__init__(**kwargs)

    def train_step(self, data):
        batch_obs, batch_acts, batch_adv, previous_log_prob = data

        with tf.GradientTape() as tape:
            result = self(batch_obs, training=True)
            curr_log_probs = tfd.Bernoulli(logits=result).log_prob(batch_acts)

            policy_ratio = tf.exp(curr_log_probs - previous_log_prob) * batch_adv
            epsilon = tf.constant(self.clip_ratio)
            limit = tf.where(batch_adv > 0, (1 + epsilon) * batch_acts, (1 - epsilon) * batch_adv)
            policy_changes = tf.math.minimum(policy_ratio, limit)

            approx_kl = tf.reduce_mean(previous_log_prob - curr_log_probs)

            loss = -tf.reduce_mean(policy_changes)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradient = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradient, trainable_vars))

        return {"loss": loss, "approx_kl": approx_kl}


def train():
    env = BigTwo()

    player_list = []
    for i in range(BigTwo.number_of_players()):
        player_list.append(RandomBot())

    env.reset()
    episode_step = 0
    while True:
        obs = env.get_current_player_obs()
        print("turn ", episode_step)
        print("current_player", obs.current_player)
        print(f"before player hand: {obs.your_hands}")
        print('last_player_played: ', obs.last_player_played)
        print('cards played: ' + Card.display_cards_string(obs.last_cards_played))

        action = player_list[obs.current_player].action(obs)
        new_obs, reward, done = env.step(action)
        episode_step += 1
        print('action: ' + str(action))
        print(f"after player hand: {new_obs.your_hands}")
        print("====")

        if done:
            env.display_all_player_hands()
            break


if __name__ == '__main__':
    train()
