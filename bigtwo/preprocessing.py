import itertools
import typing
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from bigtwo.bigtwo import BigTwo, BigTwoHand, BigTwoObservation
from playingcards.card import Card, Rank, Suit

SUIT_OHE = {
    Suit.spades: [0, 0, 0, 1],
    Suit.hearts: [0, 0, 1, 0],
    Suit.clubs: [0, 1, 0, 0],
    Suit.diamond: [1, 0, 0, 0],
}

RANK_OHE = {
    Rank.ace: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    Rank.two: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    Rank.three: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    Rank.four: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    Rank.five: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    Rank.six: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    Rank.seven: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    Rank.eight: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    Rank.nine: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    Rank.ten: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    Rank.jack: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    Rank.queen: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    Rank.king: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

EMPTY_SUIT = [0, 0, 0, 0]
EMPTY_RANK = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def create_action_cat_mapping(
    num_cards_in_hand=13,
) -> Tuple[typing.Dict[int, List[int]], typing.Dict[frozenset, int]]:
    """
    :param num_cards_in_hand:
    :return: a map of category to raw action (one hot encoded) and also a map for reverese lookup of indices to category

    {0: [0, 0, 0, 0, 0, 0]}, {(0,0 ,0,0,): 1}
    """
    result = {}
    cat = 0
    reverse_lookup = {}

    # skip action
    result[cat] = [0] * num_cards_in_hand
    cat += 1

    # play single card action
    for i in range(num_cards_in_hand):
        temp = [0] * num_cards_in_hand
        temp[i] = 1
        result[cat] = temp

        reverse_lookup[frozenset([i])] = cat
        cat += 1

    # play double card action
    for pair_idx in itertools.combinations(range(num_cards_in_hand), 2):
        temp = [0] * num_cards_in_hand
        temp[pair_idx[0]] = 1
        temp[pair_idx[1]] = 1
        reverse_lookup[frozenset(pair_idx)] = cat
        result[cat] = temp
        cat += 1

    # play combinations action
    for comb_idx in itertools.combinations(range(num_cards_in_hand), 5):
        temp = [0] * num_cards_in_hand
        reverse_lookup[frozenset(comb_idx)] = cat
        for idx in comb_idx:
            temp[idx] = 1
        result[cat] = temp
        cat += 1

    return result, reverse_lookup


def cards_to_ohe(cards: List[Card], padded_length: int) -> List[int]:
    if len(cards) > padded_length:
        raise ValueError("expected card length to be less or equal to padded length")
    data = []

    for i in range(padded_length):
        # add place holder for empty card
        if i >= len(cards):
            data += EMPTY_SUIT
            data += EMPTY_RANK
            continue

        curr_card = cards[i]
        data += SUIT_OHE[curr_card.suit]
        data += RANK_OHE[curr_card.rank]

    return data


def obs_to_arr(obs: BigTwoObservation) -> np.ndarray:
    data = []
    data += obs.num_card_per_player

    for i in range(5):
        # add place holder for empty card
        if i >= len(obs.last_cards_played):
            data += [-1, -1]
            continue

        curr_card = obs.last_cards_played[i]
        data += [Card.SUIT_NUMBER[curr_card.suit], Card.RANK_NUMBER[curr_card.rank]]

    for i in range(13):
        if i >= len(obs.your_hands):
            data += [-1, -1]
            continue

        curr_card = obs.your_hands[i]
        data += [Card.SUIT_NUMBER[curr_card.suit], Card.RANK_NUMBER[curr_card.rank]]

    return np.array(data)


def obs_to_ohe(obs: BigTwoObservation, include_last_cards_played=True) -> np.ndarray:
    data = []
    data += obs.num_card_per_player

    # first turn
    data.append(1 if len(obs.last_cards_played) == 0 else 0)

    # current player number
    current_player_number = [0, 0, 0, 0]
    current_player_number[obs.current_player] = 1

    data += current_player_number

    # last player number
    last_player_number = [0, 0, 0, 0]
    if obs.last_player_played <= 4:
        last_player_number[obs.last_player_played] = 1
    data += last_player_number

    # cards length
    data.append(len(obs.last_cards_played))

    if include_last_cards_played:
        data += cards_to_ohe(obs.last_cards_played, 5)
    data += cards_to_ohe(obs.your_hands.cards, 13)

    return np.array(data)


def generate_action_mask_first_turn(
    idx_cat_mapping,
    obs: BigTwoObservation,
) -> npt.NDArray:
    result = np.full(len(idx_cat_mapping) + 1, False)

    diamond_three = Card(Suit.diamond, Rank.three)

    card_idx_mapping = {}
    for idx in range(len(obs.your_hands)):
        card = obs.your_hands[idx]
        card_idx_mapping[card] = idx

        if card != diamond_three:
            continue

        cat = idx_cat_mapping[frozenset([idx])]
        result[cat] = True

    for pair in obs.your_hands.pairs:
        if diamond_three not in pair:
            continue

        pair_idx = [card_idx_mapping.get(card) for card in pair]
        cat = idx_cat_mapping[frozenset(pair_idx)]
        result[cat] = True

    for _, combinations in obs.your_hands.combinations.items():
        for comb in combinations:
            if diamond_three not in comb:
                continue

            comb_idx = [card_idx_mapping.get(card) for card in comb]
            cat = idx_cat_mapping[frozenset(comb_idx)]
            result[cat] = True

    return result


def generate_all_option(obs: BigTwoObservation):
    card_idx_mapping = {obs.your_hands[idx]: idx for idx in range(len(obs.your_hands))}
    single_options = [[idx] for idx in range(len(obs.your_hands))]
    pair_options = [
        [card_idx_mapping.get(card) for card in pair] for pair in obs.your_hands.pairs
    ]
    combination_options = []
    for _, combinations in obs.your_hands.combinations.items():
        for comb in combinations:
            comb_idx = [card_idx_mapping.get(card) for card in comb]
            combination_options.append(comb_idx)

    return single_options + pair_options + combination_options


def generate_single_options(hands: BigTwoHand, target_to_beat: Card) -> List[List[int]]:
    """
    :param hands:
    :param target_to_beat:
    :return: a list of indexes, each list of indexes represent the cards to play
    """
    return [
        [idx]
        for idx, card in enumerate(hands)
        if BigTwo.is_bigger([card], [target_to_beat])
    ]


def generate_pair_options(
    hands: BigTwoHand, target_to_beat: List[Card]
) -> List[List[int]]:
    """
    :param hands:
    :param target_to_beat:
    :return: a list of indexes, each list of indexes represent the cards to play
    """
    if len(target_to_beat) != 2:
        raise ValueError("expects target_to_beat to have 2 cards")

    card_idx_mapping = {hands[idx]: idx for idx in range(len(hands))}
    return [
        [card_idx_mapping[card] for card in pair]
        for pair in hands.pairs
        if BigTwo.is_bigger(list(pair), target_to_beat)
    ]


def generate_combinations_options(
    hands: BigTwoHand, target_to_beat: List[Card]
) -> List[List[int]]:
    """
    :param hands:
    :param target_to_beat:
    :return: a list of indexes, each list of indexes represent the cards to play
    """
    if len(target_to_beat) != 5:
        raise ValueError("expects target_to_beat to have 5 cards")

    card_idx_mapping = {hands[idx]: idx for idx in range(len(hands))}
    combination_options = []
    for _, combinations in hands.combinations.items():
        for comb in combinations:
            if not BigTwo.is_bigger(comb, target_to_beat):
                continue

            comb_idx = [card_idx_mapping[card] for card in comb]
            combination_options.append(comb_idx)
    return combination_options


def generate_action_mask(
    idx_cat_mapping,
    obs: BigTwoObservation,
) -> npt.NDArray:
    if obs.is_first_turn():
        return generate_action_mask_first_turn(idx_cat_mapping, obs)

    if obs.can_play_any_cards():
        options = generate_all_option(obs)
    elif len(obs.last_cards_played) == 1:
        options = generate_single_options(obs.your_hands, obs.last_cards_played[0])
    elif len(obs.last_cards_played) == 2:
        options = generate_pair_options(obs.your_hands, obs.last_cards_played)
    elif len(obs.last_cards_played) == 5:
        options = generate_combinations_options(obs.your_hands, obs.last_cards_played)
    else:
        raise ValueError("unexpected scenario to generate invalid actions mask")

    # idx_cat_mapping + 1 beacause we need to include no action
    result = np.full(len(idx_cat_mapping) + 1, False)
    result[0] = obs.can_skip()

    for option in options:
        cat = idx_cat_mapping[frozenset(option)]
        result[cat] = True

    return result
