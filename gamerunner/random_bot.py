import random

import numpy as np

from big_two_bot import BigTwoBot
from bigtwo.bigtwo import BigTwoObservation


class RandomBot(BigTwoBot):
    def action(self, observation: BigTwoObservation):
        is_last_player = not observation.last_player_played == observation.current_player
        selected_cards = np.repeat(0, 13)
        if random.randint(0, 9) < 3 and is_last_player:
            return selected_cards

        filter_hand = [x for x in observation.your_hands if not x == -1]
        random_index = random.choice(range(len(filter_hand)))
        selected_cards[random_index] = 1
        return selected_cards
