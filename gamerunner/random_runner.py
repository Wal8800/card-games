from gamerunner import RandomBot, CommandLineBot
from playingcards import Deck, Card
from bigtwo import BigTwo


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
        print("current_player", obs[3])
        print('before player hand:' + Card.display_cards_string(obs[2]))
        print('last_player_played: ', obs[4])
        print('cards played: ' + Card.display_cards_string(obs[1]))

        action = player_list[obs[3]].action(obs)
        new_obs, reward, done = env.step(action)
        episode_step += 1
        print('action: ' + str(action[1]))
        print('after player hand:' + Card.display_cards_string(new_obs[2]))
        print("====")

        if done:
            env.display_all_player_hands()
            break


if __name__ == '__main__':
    train()
