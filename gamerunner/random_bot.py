import random
from .big_two_bot import BigTwoBot


class RandomBot(BigTwoBot):
    def action(self, observation):
        is_last_player = not observation["last_player_played"] == observation["your_player_number"]
        if random.randint(0, 9) < 3 and is_last_player:
            return []

        return [random.choice(observation["your_hands"])]

