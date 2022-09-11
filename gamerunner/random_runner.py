from random_bot import RandomBot

from bigtwo.bigtwo import BigTwo


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
        print("last_player_played: ", obs.last_player_played)
        print(f"cards played: {obs.last_cards_played}")

        action = player_list[obs.current_player].action(obs)
        new_obs, done = env.step(action)
        episode_step += 1
        print("action: " + str(action))
        print(f"after player hand: {new_obs.your_hands}")
        print("====")

        if done:
            env.display_all_player_hands()
            break


if __name__ == "__main__":
    train()
