from big_two_bot import BigTwoBot
from bigtwo.bigtwo import BigTwoObservation


class CommandLineBot(BigTwoBot):
    def action(self, observation: BigTwoObservation):
        print("Your turn: ")
        print("Your hand:" + str(observation.your_hands))
        hands = input('Enter the cards you want to play (1-13) or enter "skip" : ')

        if hands == "skip":
            hand_index = []
        else:
            # assume space separated list, start from 1
            hand_index = [int(x) - 1 for x in hands.split(" ") if not x == ""]

        action = [observation.your_hands[i] for i in hand_index]
        print(f"playing: {action}")
        raw_action = [1 if x in hand_index else 0 for x in range(13)]
        print(f"playing_raw: {raw_action}")
        return raw_action
