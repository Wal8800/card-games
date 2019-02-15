import random
from .big_two_bot import BigTwoBot
import numpy as np


class RandomBot(BigTwoBot):
    def action(self, observation):
        is_last_player = not observation["last_player_played"] == observation["current_player_number"]
        selected_cards = np.repeat(0, 13)
        if random.randint(0, 9) < 3 and is_last_player:
            return (
                observation["current_player_number"],
                selected_cards
            )

        random_index = random.choice(range(len(observation["your_hands"])))
        selected_cards[random_index] = 1
        return (
            observation["current_player_number"],
            selected_cards
        )

