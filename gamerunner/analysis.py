import abc
import os
import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm

from bigtwo import bigtwo
from bigtwo.bigtwo import RANK_ORDER, BigTwoHand
from playingcards.card import Card, Deck, Rank

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 100000)

"""
question:

for single/pairs
- how much bigger than the target it plays?'
- out of the possible options, did it play the largest options?
- when it can play any cards, what's the usual thing it will play 
  - combinations
  - ranks
"""


class EpisodeProcessor(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return True

    @abc.abstractmethod
    def should_process(
        self, last_player_played: int, target: List[Card], action: List[Card]
    ) -> bool:
        raise NotImplementedError("should_process not implemented")

    @abc.abstractmethod
    def process_action(
        self, target: List[Card], current_hand: List[Card], action: List[Card]
    ):
        raise NotImplementedError("process not implemented")

    @abc.abstractmethod
    def on_episode_end(self, episode: int):
        raise NotImplementedError("on_episode_end not implemented")

    @abc.abstractmethod
    def save_data(self, output_dir: str):
        raise NotImplementedError("save_data not implemented")


class PlayAgainstOpponent(EpisodeProcessor):
    def __init__(self):
        # track across episode
        self.mean_single_rank_diff = []
        self.mean_double_rank_diff = []
        self.single_rank_dist_interval = 100
        self.single_rank_diff_comparision = list(
            range(0, 1000, self.single_rank_dist_interval)
        )
        self.single_rank_diff_dist = []

        # track within episode
        self.single_rank_diff = []
        self.double_rank_diff = []

    def should_process(
        self, last_player_played: int, target: List[Card], action: List[Card]
    ) -> bool:
        return (
            last_player_played != 0
            and (len(target) == 1 or len(target) == 2)
            and len(action) > 0
        )

    def process_action(
        self, target: List[Card], current_hand: List[Card], action: List[Card]
    ):
        target_rank = target[0].rank
        action_rank = action[0].rank
        rank_diff = RANK_ORDER[action_rank] - RANK_ORDER[target_rank]

        if len(target) == 1:
            self.single_rank_diff.append(rank_diff)

        if len(target) == 2:
            self.double_rank_diff.append(rank_diff)

    def on_episode_end(self, episode: int):
        self.mean_single_rank_diff.append(np.mean(self.single_rank_diff))
        self.mean_double_rank_diff.append(np.mean(self.double_rank_diff))

        if episode in self.single_rank_diff_comparision:
            self.single_rank_diff_dist.append(self.single_rank_diff)

        self.single_rank_diff = []
        self.double_rank_diff = []

    def save_data(self, output_dir: str):
        sns.lineplot(data=self.mean_single_rank_diff).set_title(
            "Average single card rank difference"
        )
        plt.savefig(fname=f"{output_dir}/mean_single_rank_diff.png")
        plt.clf()

        sns.lineplot(data=self.mean_double_rank_diff).set_title(
            "Average double card rank difference"
        )
        plt.savefig(fname=f"{output_dir}/mean_double_rank_diff.png")
        plt.clf()

        sns.kdeplot(data=self.single_rank_diff_dist).set_title(
            f"Distribution of single card rank difference per {self.single_rank_dist_interval} episode"
        )
        plt.savefig(fname=f"{output_dir}/mean_single_rank_dist.png")
        plt.clf()


class PlayAnyCard(EpisodeProcessor):
    def __init__(self):
        # list for tracking across episode
        self.mean_s_max_rank_diff = []
        self.mean_s_min_rank_diff = []
        self.mean_p_max_rank_diff = []
        self.mean_p_min_rank_diff = []

        self.n_action_played_history = []

        # list for tracking within an episode
        self.single_max_rank_diff = []
        self.single_min_rank_diff = []
        self.pair_max_rank_diff = []
        self.pair_min_rank_diff = []
        self.n_action_played = {}

    def should_process(
        self, last_player_played: int, target: List[Card], action: List[Card]
    ) -> bool:
        return last_player_played == 0

    def process_action(
        self, target: List[Card], current_hand: List[Card], action: List[Card]
    ):
        hand = BigTwoHand(current_hand)
        action_rank = RANK_ORDER[action[0].rank]
        if len(action) == 1:
            card_ranks = [RANK_ORDER[c.rank] for c in current_hand]
            max_diff = max(card_ranks) - action_rank
            min_diff = action_rank - min(card_ranks)

            self.single_max_rank_diff.append(max_diff)
            self.single_min_rank_diff.append(min_diff)

        if len(action) == 2:
            pair_ranks = [RANK_ORDER[pair[0].rank] for pair in hand.pairs]

            max_diff = max(pair_ranks) - action_rank
            min_diff = action_rank - min(pair_ranks)

            self.pair_max_rank_diff.append(max_diff)
            self.pair_min_rank_diff.append(min_diff)

        if len(hand.pairs) > 0 and len(hand.combinations) > 0:
            self.n_action_played[len(action)] = (
                self.n_action_played.get(len(action), 0) + 1
            )

    def on_episode_end(self, episode: int):
        self.mean_s_max_rank_diff.append(np.mean(self.single_max_rank_diff))
        self.mean_s_min_rank_diff.append(np.mean(self.single_min_rank_diff))

        self.mean_p_max_rank_diff.append(np.mean(self.pair_max_rank_diff))
        self.mean_p_min_rank_diff.append(np.mean(self.pair_min_rank_diff))

        self.n_action_played_history.append(self.n_action_played)

        self.single_max_rank_diff = []
        self.single_min_rank_diff = []
        self.pair_max_rank_diff = []
        self.pair_min_rank_diff = []
        self.n_action_played = {}

    def save_data(self, output_dir: str):
        sns.lineplot(data=self.mean_s_max_rank_diff).set_title(
            "Average single card diff compared to highest card when any card"
        )
        plt.savefig(fname=f"{output_dir}/mean_s_max.png")
        plt.clf()

        sns.lineplot(data=self.mean_s_min_rank_diff).set_title(
            "Average single card diff compared to lowest card when any card"
        )
        plt.savefig(fname=f"{output_dir}/mean_s_min.png")
        plt.clf()

        sns.lineplot(data=self.mean_p_max_rank_diff).set_title(
            "Average pair rank diff compared to highest card when any card"
        )
        plt.savefig(fname=f"{output_dir}/mean_p_max.png")
        plt.clf()

        sns.lineplot(data=self.mean_p_min_rank_diff).set_title(
            "Average pair rank diff compared to lowest card when any card"
        )
        plt.savefig(fname=f"{output_dir}/mean_p_min.png")
        plt.clf()

        for n in [1, 2, 5]:
            data = [c.get(n) for c in self.n_action_played_history]
            sns.lineplot(data=data).set_title(
                f"Number of times played {n} card when play any"
            )
            plt.savefig(fname=f"{output_dir}/{n}_cards_play_any.png")
            plt.clf()


class EpisodeProcessors:
    def __init__(self, processors: List[EpisodeProcessor]):
        self.processors = processors

    def process_action(
        self,
        last_player_played: int,
        target: List[Card],
        current_hand: List[Card],
        action: List[Card],
    ):
        for p in self.processors:
            if not p.should_process(last_player_played, target, action):
                continue

            p.process_action(target, current_hand, action)

    def on_episode_end(self, episode: int):
        for p in self.processors:
            p.on_episode_end(episode)

    def save_data(self, output_dir: str):
        for p in self.processors:
            p.save_data(output_dir)


def winning_str_distribution():
    dir_folder = "2021_07_10_10_21_42"

    data = []

    for episode in list(range(500, 1000, 100)) + [999]:
        with open(
            f"experiments/{dir_folder}/starting_hands/starting_hands_{episode}.pickle",
            "rb",
        ) as pickle_file:
            result: List[Tuple[List[List[Card]], int, int]] = pickle.load(pickle_file)

            for hands, player_won, starting_player in result:
                row = {
                    "player_won": player_won,
                    "episodes": episode,
                    "starting_player": starting_player,
                }
                for i, hand in enumerate(hands):
                    bigtwo_hand = BigTwoHand(hand)
                    row[f"player_{i}_card_strength"] = hand_strength_calculator(
                        bigtwo_hand
                    )

                    ace_count, two_count = 0, 0
                    for card in hand:
                        if card.rank == Rank.ace:
                            ace_count += 1
                            continue

                        if card.rank == Rank.two:
                            two_count += 1
                            continue
                    row[f"player_{i}_num_of_ace"] = ace_count
                    row[f"player_{i}_num_of_two"] = two_count

                data.append(row)

    df = pd.DataFrame(data=data)

    def highest_hand_strength(row_data) -> int:

        strengths = [
            row_data[f"player_{player_number}_card_strength"]
            for player_number in range(4)
        ]

        return np.argmax(np.array(strengths))

    def winning_hand_strength(row_data) -> int:
        player_won_number = row_data["player_won"]

        return row_data[f"player_{player_won_number}_card_strength"]

    def winning_hand_ace_two_count(row_data) -> int:
        player_won_number = row_data["player_won"]
        return (
            row_data[f"player_{player_won_number}_num_of_ace"]
            + row_data[f"player_{player_won_number}_num_of_two"]
        )

    df["player_with_strongest_hand"] = df.apply(highest_hand_strength, axis=1)
    df["winning_hand_str"] = df.apply(winning_hand_strength, axis=1)
    df["winning_hand_ace_two_count"] = df.apply(winning_hand_ace_two_count, axis=1)

    temp = df[df["player_0_card_strength"] >= 110]
    print(temp["player_won"].value_counts())
    # sns.displot(data=df, x="player_0_card_strength")
    plt.show()


def analyse_starting_hands():
    dir_folder = "2021_06_26_13_58_47_serialise"

    data = {
        "episodes": [],
        "player_0_card_strength": [],
        "player_1_card_strength": [],
        "player_2_card_strength": [],
        "player_3_card_strength": [],
        "player_won": [],
        "starting_player": [],
    }

    for episode in range(0, 10, 1):
        with open(
            f"starting_hands/{dir_folder}/starting_hands_{episode}.pickle", "rb"
        ) as pickle_file:
            result: List[Tuple[List[List[Card]], int, int]] = pickle.load(pickle_file)

            for hands, player_won, starting_player in result:
                data.get("player_won", []).append(player_won)
                data.get("episodes", []).append(episode)
                data.get("starting_player", []).append(starting_player)

                for i, hand in enumerate(hands):
                    bigtwo_hand = BigTwoHand(hand)
                    data.get(f"player_{i}_card_strength", []).append(
                        hand_strength_calculator(bigtwo_hand)
                    )

        break

    df = pd.DataFrame(data=data)

    def highest_hand_strength(row) -> int:

        strengths = [
            row[f"player_{player_number}_card_strength"] for player_number in range(4)
        ]

        return np.argmax(np.array(strengths))

    def winning_hand_strength(row) -> int:
        player_won_number = row["player_won"]

        return row[f"player_{player_won_number}_card_strength"]

    df["player_with_strongest_hand"] = df.apply(highest_hand_strength, axis=1)
    df["winning_hand_str"] = df.apply(winning_hand_strength, axis=1)

    print(df.shape)
    print(df.head())

    print(df[df.winning_hand_str < 80])

    sns.displot(data=df["winning_hand_str"])
    plt.show()


def main():
    dir_folder = "2021_06_19_23_09_06"

    processors = EpisodeProcessors([PlayAnyCard(), PlayAgainstOpponent()])

    for episode in tqdm.tqdm(range(0, 5000, 10)):
        with open(
            f"action_history/{dir_folder}/action_played_{episode}.pickle", "rb"
        ) as pickle_file:
            result: List[Tuple[int, List[Card], List[Card], List[Card]]] = pickle.load(
                pickle_file
            )

            for last_player_played, target, current_hand, action in result:
                processors.process_action(
                    last_player_played, target, current_hand, action
                )

            processors.on_episode_end(episode)

    output_root_dir = "analysis_result"
    output_dir = f"{output_root_dir}/{dir_folder}"
    os.makedirs(output_dir, exist_ok=True)

    processors.save_data(output_dir)


def hand_strength_calculator(player_hand: BigTwoHand) -> int:
    strength = sum([bigtwo.RANK_ORDER[card.rank] for card in player_hand])

    # for card_one, _ in player_hand.pairs:
    #     strength += bigtwo.rank_order[card_one.rank]
    #
    # for combination_types in player_hand.combinations:
    #     if combination_types == BigTwo.STRAIGHT:
    #         strength += 10
    #     elif combination_types == BigTwo.FLUSH:
    #         strength += 20
    #     elif combination_types == BigTwo.FULL_HOUSE:
    #         strength += 30
    #     elif combination_types == BigTwo.FOUR_OF_A_KIND:
    #         strength += 40
    #     elif combination_types == BigTwo.STRAIGHT_FLUSH:
    #         strength += 50

    return strength


def hand_strength():

    strength_distribution = []
    for _ in range(40000):
        hands = Deck().shuffle_and_split(4)
        player_hand = BigTwoHand(hands[0])
        strength_distribution.append(hand_strength_calculator(player_hand))

    sns.displot(data=strength_distribution)
    plt.show()


def hand_strength_for_winning_hands():
    dir_folder = "2021_06_19_23_09_06"
    count = 0
    for episode in tqdm.tqdm(range(0, 5000, 10)):
        with open(
            f"action_history/{dir_folder}/action_played_{episode}.pickle", "rb"
        ) as pickle_file:
            result: List[Tuple[int, List[Card], List[Card], List[Card]]] = pickle.load(
                pickle_file
            )

            for last_player_played, target, current_hand, action in result:
                if len(current_hand) == len(action):
                    count += 1

    print(count)


if __name__ == "__main__":
    winning_str_distribution()
