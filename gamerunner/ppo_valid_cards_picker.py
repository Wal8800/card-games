from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from bigtwo.valid_cards_game import ValidCardGame, pick_cards_from_hand
from playingcards.card import Card


base_env = ValidCardGame()
env = DummyVecEnv([lambda: base_env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000)

obs = env.reset()
total_rewards = 0
for i in range(20):
    action, _states = model.predict(obs)
    new_obs, rewards, dones, info = env.step(action)

    if rewards[0] > 0:
        hand = [Card.from_number(x) for x in obs[0]]
        cards = pick_cards_from_hand(action[0], hand)
        hand_str = Card.display_cards_string(obs[0])
        card_str = ' '.join(str(x) for x in cards)
        print("hand: {:s} selected: {:s} reward: {:f}".format(hand_str, card_str, rewards[0]))

    obs = new_obs
    total_rewards += rewards[0]
    # env.render()


print("total rewards: ", total_rewards)
# env.env_method("display_past_result")
