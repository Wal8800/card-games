import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from bigtwo.valid_cards_game import ValidCardGame
from playingcards.card import Card


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network.
    for size in sizes[:-1]:
        x = tf.layers.dense(x, units=size, activation=activation)
    return tf.layers.dense(x, units=sizes[-1], activation=output_activation)


def random_hand():
    return np.random.randint(52, size=13)


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


obs_dim = 13
n_acts = 13
hidden_size = 32
batch_size=5000

# core of the policy network
obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32, name="obs_ph")
logits = mlp(obs_ph, [32, n_acts])

# make action selection op (outputs int actions, sampled from policy)
# actions = tf.squeeze(tf.multinomial(logits=logits, num_samples=1), axis=1)
actions = tfp.distributions.Bernoulli(logits=logits).sample()

# make loss function whose gradient, for the right data, is policy gradient
weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name="weights_ph")
act_ph = tf.placeholder(shape=(None, n_acts), dtype=tf.float32, name="act_ph")
log_probs = tf.reduce_sum(act_ph * tf.nn.log_softmax(logits), axis=1)
loss = -tf.reduce_mean(weights_ph * log_probs)

# make train_op
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


def train_one_epoch():
    env = ValidCardGame()
    obs = env.reset()
    ep_rews = []

    # make some empty lists for logging.
    batch_obs = []  # for observations
    batch_acts = []  # for actions
    batch_weights = []  # for R(tau) weighting in policy gradient
    batch_rets = []  # for measuring episode returns
    batch_lens = []  # for measuring episode lengths

    while True:
        batch_obs.append(obs.copy())
        action = sess.run(actions, {obs_ph: obs.reshape(1, -1)})[0]

        obs, reward, done = env.step(action)

        ep_rews.append(reward)
        batch_acts.append(action)

        if done:
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # the weight for each logprob(a|s) is R(tau)
            batch_weights += list(reward_to_go(ep_rews))

            # reset episode-specific variables
            obs, done, ep_rews = env.reset(), False, []

            # end experience loop if we have enough of it
            if len(batch_obs) > batch_size:
                break

    batch_obs_arr =  np.array(batch_obs)
    batch_act_arr =  np.array(batch_acts)
    batch_weight_arr = np.array(batch_weights)
    batch_loss, _ = sess.run([loss, train_op],
                             feed_dict={
                                 obs_ph: batch_obs_arr,
                                 act_ph: batch_act_arr,
                                 weights_ph: batch_weight_arr
                             })
    return batch_loss, batch_rets, batch_lens


for j in range(50):
    batch_loss, batch_rets, batch_lens = train_one_epoch()
    print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' % (j, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))