from .big_two_bot import BigTwoBot


class CommandLineBot(BigTwoBot):
    def action(self, observation):
        print("Your turn: ", self.name)
        print('Your hand:' + ' '.join(str(x) for x in observation.your_hands))
        hands = input('Enter the cards you want to play (1-13) or enter "skip" : ')

        if hands == "skip":
            return []

        # assume space separated list, start from 1
        hand_index = [int(x)-1 for x in hands.split(" ") if not x == ""]
        return [observation.your_hands[i] for i in hand_index]
