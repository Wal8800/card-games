from gamerunner import Bot
from playingcards import Deck
from bigtwo import BigTwo


deck = Deck()
player_list = []
for i in range(BigTwo.number_of_players()):
    player_list.append(Bot(str(i)))

big_two_game = BigTwo(player_list, deck)

turn = 0

while True:
    current_player, current_player_number = big_two_game.get_current_player()

    obs = big_two_game.current_observation(current_player_number)
    action = current_player.action(obs, obs.last_player_played == current_player_number)
    result_obs, reward, done = big_two_game.step(action)
    print("turn ", turn)
    print("current_player", current_player_number)
    print('before player hand:' + ' '.join(str(x) for x in obs.your_hands))
    print('last_player_played: ', obs.last_player_played)
    print('cards played: ' + str(obs))
    print('action: ' + ' '.join(str(x) for x in action))
    print('after player hand:' + ' '.join(str(x) for x in result_obs.your_hands))
    print("====")
    if done:
        break

    turn += 1

big_two_game.display_all_player_hands()

"""
One Level abstraction for a single bot to play the game
One lower level abstraction for all 4 player to play the game (working on this first)
"""