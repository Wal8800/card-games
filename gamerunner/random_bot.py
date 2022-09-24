import random
from typing import List

import numpy as np

from bigtwo.bigtwo import BigTwoObservation
from bigtwo.preprocessing import (
    generate_all_option,
    generate_combinations_options,
    generate_pair_options,
    generate_single_options,
)
from gamerunner.big_two_bot import BigTwoBot
from playingcards.card import Card, Rank, Suit


class RandomBot(BigTwoBot):
    def action(self, obs: BigTwoObservation):
        # random bot always play diamond 3
        if obs.is_first_turn():
            selected_cards = np.repeat(0, 13)
            for idx, card in enumerate(obs.your_hands):
                if card != Card(Suit.diamond, Rank.three):
                    continue

                selected_cards[idx] = 1
                return selected_cards

        options: List[List[int]]
        if obs.can_play_any_cards():
            options = generate_all_option(obs)
        elif len(obs.last_cards_played) == 1:
            options = generate_single_options(obs.your_hands, obs.last_cards_played[0])
        elif len(obs.last_cards_played) == 2:
            options = generate_pair_options(obs.your_hands, obs.last_cards_played)
        elif len(obs.last_cards_played) == 5:
            options = generate_combinations_options(
                obs.your_hands, obs.last_cards_played
            )
        else:
            raise ValueError("unexpected scenario to generate invalid actions mask")

        selected_cards = np.repeat(0, 13)
        if len(options) == 0:
            return selected_cards

        random_opt_idx = random.choice(range(len(options)))

        for idx in options[random_opt_idx]:
            selected_cards[idx] = 1
        return selected_cards
