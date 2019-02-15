from gamerunner import RandomBot, CommandLineBot
from playingcards import Deck, Card
from bigtwo import BigTwo

player_list = []
for i in range(BigTwo.number_of_players()):
    player_list.append(RandomBot())

big_two_game = BigTwo()

turn = 0


def display_cards_string(cards):
    filtered_list = [x for x in cards if x > -1]
    return ' '.join(str(Card.from_number(x)) for x in filtered_list)


obs = big_two_game.reset()

while True:
    print("turn ", turn)
    print("current_player", obs["current_player_number"])
    print('before player hand:' + display_cards_string(obs["your_hands"]))
    print('last_player_played: ', obs["last_player_played"])
    print('cards played: ' + display_cards_string(obs["last_cards_played"]))

    action = player_list[obs["current_player_number"]].action(obs)
    obs, reward, done = big_two_game.step(action)

    print('action: ' + str(action[1]))
    print('after player hand:' + display_cards_string(obs["your_hands"]))
    print("====")
    if done:
        break

    turn += 1

big_two_game.display_all_player_hands()

"""
One Level abstraction for a single bot to play the game
One lower level abstraction for all 4 player to play the game (working on this first)
"""