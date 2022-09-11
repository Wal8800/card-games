import abc
from collections import Mapping

import numpy as np
import scipy.signal
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models, optimizers
from tensorflow_probability import distributions as tfd


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


# Buffer to store tracjectories and calculate the Generalized advantage estimation
# Heaviliy inspired from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/vpg.py
class VPGBuffer:
    def __init__(self, obs_dim, size: int, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew):
        assert self.ptr < self.max_size

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew

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
        self.adv_buf[path_slice] = discounted_sum_of_rewards(
            deltas, self.gamma * self.lam
        )

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discounted_sum_of_rewards(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        return self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf


class VPGAgent:
    """
    VPG Agent for discrete observation space and discrete action space
    """

    def __init__(self, obs_dim, act_dim, dir_path: str = None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # create policy and value function from saved files.
        if dir_path is not None:
            custom_obj = {"VanillaPG": VanillaPG}
            self.policy = models.load_model(
                f"{dir_path}/policy.h5", custom_objects=custom_obj
            )
            self.value_function = models.load_model(
                f"{dir_path}/value_function.h5", custom_objects=custom_obj
            )
            return

        # creating the policy
        obs_inp = tf.keras.Input(shape=(obs_dim,), name="obs")
        x = layers.Dense(32, activation="relu")(obs_inp)
        output = layers.Dense(act_dim)(x)
        self.policy = VanillaPG(inputs=obs_inp, outputs=output)
        self.policy.compile(optimizer=optimizers.Adam(learning_rate=0.01))

        value_input = layers.Input(shape=(obs_dim,), name="obs")
        x = layers.Dense(64, activation="relu")(value_input)
        x = layers.Dense(64, activation="relu")(x)
        value_output = layers.Dense(1)(x)
        self.value_function = keras.Model(inputs=value_input, outputs=value_output)
        self.value_function.compile(
            optimizer=optimizers.Adam(0.01), loss=keras.losses.MeanSquaredError()
        )

    def action(self, observation):
        dist = tfd.Categorical(logits=self.policy(np.array([observation])))
        return dist.sample().numpy()[0]

    def predict_value(self, observations):
        return self.value_function.predict(observations)

    def update(self, buf: VPGBuffer):
        obs_buf, act_buf, adv_buf, ret_buf = buf.get()

        # update the value function
        for _ in range(80):
            value_loss = self.value_function.train_step((obs_buf, tf.constant(ret_buf)))

        # update the policy by a single step
        result = self.policy.train_step([obs_buf, act_buf, adv_buf])

        return result["loss"].numpy(), value_loss["loss"].numpy()

    def save(self, dir_path: str):
        self.policy.save(f"{dir_path}/policy.h5")
        self.value_function.save(f"{dir_path}/value_function.h5")


class PPOBufferInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "get") and callable(subclass.get)

    @abc.abstractmethod
    def get(self):
        """Get the data in the ppo buffer"""
        raise NotImplementedError


class PPOBuffer(PPOBufferInterface):
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
        self.adv_buf[path_slice] = discounted_sum_of_rewards(
            deltas, self.gamma * self.lam
        )

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discounted_sum_of_rewards(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        return self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf


def action_with_mask_x(model, obs, n_action, mask):
    logits = model(obs, training=False)

    adder = (1.0 - tf.cast(mask, logits.dtype)) * -1e9
    """
    1 = 0
    0 = -1e9
    """
    logits = logits + adder

    logp_all = tf.nn.log_softmax(logits)

    sample_action = tf.squeeze(tf.random.categorical(logits, 1))
    logp_pi = tf.squeeze(
        tf.reduce_sum(tf.one_hot(sample_action, depth=n_action) * logp_all, axis=1)
    )
    return sample_action, logp_pi


class SavedPPOAgent:
    def __init__(
        self,
        env_name: str,
        n_action: int,
        dir_path: str,
    ):
        self.env_name = env_name
        self.n_action = n_action

        self.policy = models.load_model(f"{dir_path}/{self.env_name}_policy")
        self.vf = models.load_model(f"{dir_path}/{self.env_name}_vf")

    def action(self, obs, mask=None):
        if isinstance(obs, Mapping):
            wrapped = {k: tf.constant(v) for k, v in obs.items()}
        else:
            wrapped = tf.constant(obs)
        act_dim = tf.constant(self.n_action)

        if mask is None:
            return self._action(wrapped, act_dim)

        return self._action_with_mask(wrapped, act_dim, tf.constant(mask))

    @tf.function
    def _action(self, obs, n_action):
        logits = self.policy(obs)
        logp_all = tf.nn.log_softmax(logits)

        sample_action = tf.squeeze(tf.random.categorical(logits, 1))
        logp_pi = tf.squeeze(
            tf.reduce_sum(tf.one_hot(sample_action, depth=n_action) * logp_all, axis=1)
        )
        return sample_action, logp_pi

    def _action_with_mask(self, obs, n_action, mask):
        return action_with_mask_x(self.policy, obs, n_action, mask)


class PPOAgent:
    """
    Discrete Action only
    """

    def __init__(
        self,
        policy_network: keras.Model,
        vf_network: keras.Model,
        env_name: str,
        n_action: int,
        dir_path: str = None,
        policy_lr=0.01,
        value_lr=0.01,
        clip_ratio=0.3,
    ):
        self.env_name = env_name
        self.train_v_iters = 80
        self.train_p_iters = 80
        self.target_kl = 0.01
        self.clip_ratio = clip_ratio
        self.n_action = n_action

        # create policy and value function from saved files.
        if dir_path is not None:
            self.policy = models.load_model(f"{dir_path}/{self.env_name}_policy")
            self.vf = models.load_model(f"{dir_path}/{self.env_name}_vf")
            return

        self.policy = policy_network
        self.polcy_optimizer = optimizers.Adam(learning_rate=policy_lr)

        self.vf = vf_network
        self.vf.compile(
            optimizer=optimizers.Adam(value_lr), loss=keras.losses.MeanSquaredError()
        )

    def action(self, obs, mask=None):
        if isinstance(obs, Mapping):
            wrapped = {k: tf.constant(v) for k, v in obs.items()}
        else:
            wrapped = tf.constant(obs)
        act_dim = tf.constant(self.n_action)

        if mask is None:
            return self.__action(wrapped, act_dim)

        return self.__action_with_mask(wrapped, act_dim, tf.constant(mask))

    @tf.function
    def __action(self, obs, n_action):
        logits = self.policy(obs)
        logp_all = tf.nn.log_softmax(logits)

        sample_action = tf.squeeze(tf.random.categorical(logits, 1))
        logp_pi = tf.squeeze(
            tf.reduce_sum(tf.one_hot(sample_action, depth=n_action) * logp_all, axis=1)
        )
        return sample_action, logp_pi

    def __action_with_mask(self, obs, n_action, mask):
        return action_with_mask_x(self.policy, obs, n_action, mask)

    def update(self, buf: PPOBufferInterface, mini_batch_size=32, mask_buf=None):
        obs_buf, act_buf, adv_buf, ret_buf, logp_buf = buf.get()

        # update the value function
        v_result: tf.keras.callbacks.History = self.vf.fit(
            obs_buf,
            ret_buf,
            shuffle=True,
            batch_size=mini_batch_size,
            verbose=0,
            epochs=self.train_v_iters,
        )
        value_loss = v_result.history.get("loss")[0]

        # update the policy by a single step
        inds = np.arange(len(act_buf))
        train_p_iters_loss = []
        for i in range(self.train_p_iters):
            # Randomize the indexes
            np.random.shuffle(inds)

            mb_loss = []
            mb_kl = []
            for start in range(0, len(act_buf), mini_batch_size):
                end = start + mini_batch_size
                mbinds = inds[start:end]
                slices = []
                for arr in (obs_buf, act_buf, adv_buf, logp_buf):
                    chunk = (
                        {k: v[mbinds] for k, v in arr.items()}
                        if isinstance(arr, Mapping)
                        else arr[mbinds]
                    )
                    slices.append(chunk)

                mask = mask_buf[mbinds] if mask_buf is not None else None
                p_result = self._update_policy_network(slices, mask=mask)

                mb_loss.append(p_result["loss"])
                mb_kl.append(p_result["approx_kl"])

            train_p_iters_loss.append(np.mean(mb_loss))

            avg_kl = np.mean(mb_kl)
            if avg_kl > 1.5 * self.target_kl:
                print(f"Early stopping at step {i} due to reaching max kl: {avg_kl}")
                break

        return np.mean(train_p_iters_loss), value_loss

    @tf.function
    def __calculate_loss(
        self,
        n_action,
        batch_obs,
        batch_acts,
        previous_log_prob,
        batch_adv,
        epsilon,
        mask=None,
    ):
        logits = self.policy(batch_obs, training=True)
        if mask is not None:
            adder = (1.0 - tf.cast(mask, logits.dtype)) * -1e9
            logits = logits + adder

        # logp_all = tf.nn.log_softmax(logits)
        # curr_log_probs = tf.reduce_sum(
        #     tf.one_hot(batch_acts, depth=n_action) * logp_all, axis=1
        # )

        curr_log_probs = tfd.Categorical(logits=logits).log_prob(batch_acts)

        policy_ratio = tf.exp(curr_log_probs - previous_log_prob) * batch_adv
        limit = tf.where(
            batch_adv > 0, (1 + epsilon) * batch_adv, (1 - epsilon) * batch_adv
        )
        policy_changes = tf.math.minimum(policy_ratio, limit)

        approx_kl = tf.reduce_mean(previous_log_prob - curr_log_probs)
        loss = -tf.reduce_mean(policy_changes)

        return loss, approx_kl

    def _update_policy_network(self, data, mask=None):
        batch_obs, batch_acts, batch_adv, previous_log_prob = data

        obs = (
            {k: tf.constant(v) for k, v in batch_obs.items()}
            if isinstance(batch_obs, Mapping)
            else tf.constant(batch_obs)
        )

        # normalize the advantage function value
        # make use of a costant baseline at all timesteps for all tracjectories, which does not change the policy
        # gradient.
        batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)
        with tf.GradientTape() as tape:
            # TODO investigate why the softmax with masking doesn't calculate graident correctly
            loss, approx_kl = self.__calculate_loss(
                n_action=tf.constant(self.n_action),
                batch_obs=obs,
                batch_acts=tf.constant(batch_acts, dtype=tf.int64),
                previous_log_prob=tf.constant(previous_log_prob),
                batch_adv=tf.constant(batch_adv),
                epsilon=tf.constant(self.clip_ratio),
                mask=tf.constant(mask) if mask is not None else None,
            )

        # Compute gradients
        trainable_vars = self.policy.trainable_variables
        gradient = tape.gradient(loss, trainable_vars)

        # Update weights
        self.polcy_optimizer.apply_gradients(zip(gradient, trainable_vars))

        return {"loss": loss, "approx_kl": approx_kl}

    def predict_value(self, observations):
        if isinstance(observations, Mapping):
            observations = {k: tf.constant(v) for k, v in observations.items()}
        return self.vf(observations)

    def save(self, dir_path: str):
        self.policy.save(f"{dir_path}/{self.env_name}_policy")
        self.vf.save(f"{dir_path}/{self.env_name}_vf")

    def prob(self, observation, action_cat):
        wrapped = np.array([observation])
        result = self.policy(wrapped, training=True)

        return tfd.Categorical(logits=result).prob(action_cat).numpy()[0]

    def get_weights(self):
        return self.policy.get_weights(), self.vf.get_weights()

    def set_weights(self, policy_weights, value_weights):
        self.policy.set_weights(policy_weights)
        self.vf.set_weights(value_weights)


class VanillaPG(keras.Model):
    def train_step(self, data):
        batch_obs, batch_acts, batch_weights = data

        with tf.GradientTape() as tape:
            result = self(batch_obs, training=True)

            # The output of softargmax function can be used to represent a categorical distriubtion ie a probability
            # distribution over K different possible outcomes.
            log_probs = tfd.Categorical(logits=result).log_prob(batch_acts)
            loss = -tf.reduce_mean(batch_weights * log_probs)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradient = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradient, trainable_vars))

        return {"loss": loss}


def get_mlp_policy(obs_shape, act_dim, hidden_units=64) -> keras.Model:
    obs_inp = layers.Input(shape=obs_shape, name="obs")
    x = layers.Dense(hidden_units, activation="relu")(obs_inp)
    x = layers.Dense(hidden_units, activation="relu")(x)
    output = layers.Dense(act_dim)(x)

    return keras.Model(inputs=obs_inp, outputs=output)


def get_mlp_vf(obs_shape, hidden_units=64) -> keras.Model:
    obs_inp = layers.Input(shape=obs_shape, name="obs")
    x = layers.Dense(hidden_units, activation="relu")(obs_inp)
    x = layers.Dense(hidden_units, activation="relu")(x)
    output = layers.Dense(1)(x)

    return keras.Model(inputs=obs_inp, outputs=output)


def get_2d_conv_policy(obs_shape, act_dim) -> keras.Model:
    obs_inp = layers.Input(shape=obs_shape, name="obs")
    x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(obs_inp)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(act_dim)(x)

    return keras.Model(inputs=obs_inp, outputs=output)


def get_2d_conv_vf(obs_shape) -> keras.Model:
    obs_inp = layers.Input(shape=obs_shape, name="obs")
    x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(obs_inp)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1)(x)

    return keras.Model(inputs=obs_inp, outputs=output)


def get_mlp_lstm_policy(obs_shape, act_dim) -> keras.Model:
    """
    :param obs_shape: [timesteps, feature]
    :param act_dim:
    :return:
    """
    obs_inp = layers.Input(shape=obs_shape, name="obs")
    x = layers.LSTM(10)(obs_inp)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(act_dim)(x)
    return keras.Model(inputs=obs_inp, outputs=output)


def get_mlp_lstm_value(obs_shape) -> keras.Model:
    """
    :param obs_shape: [timesteps, feature]
    :param act_dim:
    :return:
    """
    obs_inp = layers.Input(shape=obs_shape, name="obs")
    x = layers.LSTM(10)(obs_inp)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(1)(x)
    return keras.Model(inputs=obs_inp, outputs=output)
