from .bot import Bot
from playingcards import Deck
from bigtwo import BigTwo


deck = Deck()
player_list = []
for i in range(BigTwo.number_of_players()):
    player_list.append(Bot(str(i)))

big_two_game = BigTwo(player_list, deck)

while True:
    current_player, current_player_number = big_two_game.get_current_player()
    obs = big_two_game.current_observation(current_player_number)
    action = current_player.action(obs)

    _, reward, done = big_two_game.step(action)

    if done:
        break

big_two_game.display_all_player_hands()

"""
One Level abstraction for a single bot to play the game
One lower level abstraction for all 4 player to play the game (working on this first)
"""