from gamerunner import RandomBot, CommandLineBot
from playingcards import Deck, Card
from bigtwo import BigTwo, BigTwoRealTime


def train():
    env = BigTwoRealTime()

    player_list = []
    for i in range(BigTwo.number_of_players()):
        player_list.append(RandomBot())

    obs_n = env.reset()
    episode_step = 0
    while True:
        action_n = [agent.action(obs) for agent, obs in zip(player_list, obs_n)]
        new_obs_n, rew_n, done_n = env.step(action_n)
        episode_step += 1
        obs_n = new_obs_n

        if all(done_n):
            env.display_all_player_hands()
            break


if __name__ == '__main__':
    train()
