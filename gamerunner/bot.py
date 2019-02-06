import random


class Bot:
    def __init__(self, name):
        self.name = name

    def action(self, observation, no_skip=False):
        if random.randint(0, 9) < 3 and not no_skip:
            return []

        return [random.choice(observation.your_hands)]

