import random
from .big_two_bot import BigTwoBot
import numpy as np


class RandomBot(BigTwoBot):
    def action(self, observation):
        is_last_player = not observation[4] == observation[3]
        selected_cards = np.repeat(0, 13)
        if random.randint(0, 9) < 3 and is_last_player:
            return selected_cards

        filter_hand = [x for x in observation[2] if not x == -1]
        random_index = random.choice(range(len(filter_hand)))
        selected_cards[random_index] = 1
        return selected_cards

