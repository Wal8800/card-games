import itertools
from collections import Counter
from collections.abc import MutableSequence
from typing import List, Dict, Tuple

from gym import spaces

from playingcards.card import Card, Deck, Suit, Rank

straight_rank_order = {
    Rank.ace: 1,
    Rank.two: 2,
    Rank.three: 3,
    Rank.four: 4,
    Rank.five: 5,
    Rank.six: 6,
    Rank.seven: 7,
    Rank.eight: 8,
    Rank.nine: 9,
    Rank.ten: 10,
    Rank.jack: 11,
    Rank.queen: 12,
    Rank.king: 13,
}

rank_order = {
    Rank.three: 1,
    Rank.four: 2,
    Rank.five: 3,
    Rank.six: 4,
    Rank.seven: 5,
    Rank.eight: 6,
    Rank.nine: 7,
    Rank.ten: 8,
    Rank.jack: 9,
    Rank.queen: 10,
    Rank.king: 11,
    Rank.ace: 12,
    Rank.two: 13,
}

suit_order = {Suit.diamond: 1, Suit.clubs: 2, Suit.hearts: 3, Suit.spades: 4}


def have_diamond_three(hand):
    for card in hand:
        if card.suit == Suit.diamond and card.rank == Rank.three:
            return True

    return False


def find_combinations_from_cards(input_cards: List[Card]):
    """
    rank_map
        - FOUR OF A KIND
        - FULL HOUSE
        - STRAIGHT
    suit_map
        - flush
        - straight flush
    """
    rank_map: Dict[Rank, List[Card]] = {}
    suit_map: Dict[Suit, List[Card]] = {}

    straight_flush_set = set()

    for card in input_cards:
        rank_vals = rank_map.get(card.rank, [])
        rank_vals.append(card)
        rank_map[card.rank] = rank_vals

        suit_vals = suit_map.get(card.suit, [])
        suit_vals.append(card)
        suit_map[card.suit] = suit_vals

    combinations: Dict[int, List[Tuple[Card]]] = {}
    for _, cards in suit_map.items():
        if len(cards) < 5:
            continue

        for flush in itertools.combinations(cards, 5):
            c_type = BigTwo.FLUSH
            if BigTwo.is_straight(list(flush)):
                c_type = BigTwo.STRAIGHT_FLUSH
                straight_flush_set.add(frozenset(flush))

            current_list = combinations.get(c_type, [])
            current_list.append(flush)
            combinations[c_type] = current_list

    for rank, cards in rank_map.items():
        if len(cards) < 3:
            continue

        if len(cards) == 3:
            full_house = build_full_house_combinations(cards, rank, rank_map)
            current_list = combinations.get(BigTwo.FULL_HOUSE, []) + full_house
            combinations[BigTwo.FULL_HOUSE] = current_list
            continue

        # have all 4 cards
        # first create four of a kind
        for card in input_cards:
            if card.rank == rank:
                continue

            four_of_a_kind = cards + [card]
            current_list = combinations.get(BigTwo.FOUR_OF_A_KIND, [])
            current_list.append(tuple(four_of_a_kind))
            combinations[BigTwo.FOUR_OF_A_KIND] = current_list

        # then build full house
        for three_of_kind in itertools.combinations(cards, 3):
            full_house = build_full_house_combinations(
                list(three_of_kind), rank, rank_map
            )
            current_list = combinations.get(BigTwo.FULL_HOUSE, []) + full_house
            combinations[BigTwo.FULL_HOUSE] = current_list

    sorted_ranks = sorted(list(rank_map.keys()), key=lambda r: straight_rank_order[r])

    if len(sorted_ranks) < 5:
        # can't have straight if we don't have at least 5 unique rank
        return combinations

    # scrolling window of 5 cards to check for straights
    # assuming if the last rank and first rank have distance of 4 cards then
    # whatever inside the window is valid straights This is because we sorted the list of rank by order
    for i in range(0, len(sorted_ranks) - 4):
        first_rank = sorted_ranks[i]
        last_rank = sorted_ranks[i + 4]
        if straight_rank_order[first_rank] + 4 != straight_rank_order[last_rank]:
            continue

        list_of_rank_cards = [rank_map[rank] for rank in sorted_ranks[i : i + 5]]

        straights = list(itertools.product(*list_of_rank_cards))
        straights = [s for s in straights if frozenset(s) not in straight_flush_set]

        current_list = combinations.get(BigTwo.STRAIGHT, []) + straights
        combinations[BigTwo.STRAIGHT] = current_list

    # check for special case ten to ace straight
    if sorted_ranks[-4] == Rank.ten and sorted_ranks[0] == Rank.ace:
        list_of_rank_cards = [
            rank_map[Rank.ten],
            rank_map[Rank.jack],
            rank_map[Rank.queen],
            rank_map[Rank.king],
            rank_map[Rank.ace],
        ]
        straights = itertools.product(*list_of_rank_cards)
        straights = [s for s in straights if frozenset(s) not in straight_flush_set]

        current_list = combinations.get(BigTwo.STRAIGHT, []) + straights
        combinations[BigTwo.STRAIGHT] = current_list

    if BigTwo.STRAIGHT in combinations and len(combinations[BigTwo.STRAIGHT]) == 0:
        del combinations[BigTwo.STRAIGHT]

    return combinations


def build_full_house_combinations(
    three_of_a_kinds: List[Card], three_rank: Rank, rank_map: Dict[Rank, List[Card]]
):
    result = []
    for pair_rank, pair_cards in rank_map.items():
        # need at least 2 cards
        if len(pair_cards) < 2:
            continue

        # can't use the same rank to create full house
        if three_rank == pair_rank:
            continue

        for full_house_pair in itertools.combinations(pair_cards, 2):
            full_house = three_of_a_kinds + list(full_house_pair)
            result.append(tuple(full_house))

    return result


def bf_find_combinations_from_cards(
    input_cards: List[Card],
) -> Dict[int, List[Tuple[Card]]]:
    combinations: Dict[int, List[Tuple[Card]]] = {}
    for combination in itertools.combinations(input_cards, 5):
        try:
            c_type = BigTwo.get_five_card_type(list(combination))

            current_list = combinations.get(c_type, [])
            current_list.append(combination)
            combinations[c_type] = current_list
        except ValueError:
            continue

    return combinations


class BigTwoHand(MutableSequence):
    def __init__(self, cards: List[Card]):
        self.cards = cards
        self.pairs: List[Tuple[Card]] = []
        for pair in itertools.combinations(self.cards, 2):
            if BigTwo.is_valid_card_combination(list(pair)):
                self.pairs.append(pair)

        self.combinations = find_combinations_from_cards(self.cards)

        super().__init__()

    def have_diamond_three(self):
        for card in self.cards:
            if card.suit == Suit.diamond and card.rank == Rank.three:
                return True

        return False

    def remove_cards(self, to_remove: List[Card]):
        for card in to_remove:
            self.cards.remove(card)
            self.__remove_combinations(card)
            self.__remove_pairs(card)

    def __remove_pairs(self, target: Card):
        self.pairs = [pair for pair in self.pairs if target not in pair]

    def __remove_combinations(self, target: Card):
        c_types = list(self.combinations.keys())
        for c_type in c_types:
            combinations = self.combinations[c_type]
            new_combinations = [x for x in combinations if target not in x]

            if len(new_combinations) > 0:
                self.combinations[c_type] = new_combinations
            else:
                del self.combinations[c_type]

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, i):
        return self.cards[i]

    def __delitem__(self, key):
        del self.cards[key]

    def __setitem__(self, key, value):
        self.cards[key] = value

    def insert(self, index: int, value: Card) -> None:
        self.cards.insert(index, value)

    def __repr__(self):
        return " ".join([str(c) for c in self.cards])

    def is_full(self):
        return len(self.cards) == 13


class BigTwoObservation:
    def __init__(
        self,
        num_card_per_player: List[int],
        last_cards_played: List[Card],
        your_hands: BigTwoHand,
        current_player: int,
        last_player_played: int,
    ):
        self.num_card_per_player: List[int] = num_card_per_player
        self.last_cards_played: List[Card] = last_cards_played
        self.your_hands: BigTwoHand = your_hands
        self.current_player: int = current_player
        self.last_player_played: int = last_player_played

    def can_skip(self) -> bool:
        return (
            self.last_player_played != BigTwo.UNKNOWN_PLAYER
            and self.last_player_played != self.current_player
        )

    def is_first_turn(self) -> bool:
        return self.last_player_played == BigTwo.UNKNOWN_PLAYER

    def can_play_any_cards(self) -> bool:
        return (
            len(self.last_cards_played) == 0
            or self.last_player_played == self.current_player
        )


class BigTwo:
    FULL_HOUSE = 1
    FOUR_OF_A_KIND = 2
    STRAIGHT = 3
    FLUSH = 4
    STRAIGHT_FLUSH = 5

    UNKNOWN_PLAYER = 5

    INVALID_MOVE_REWARD = 0

    def __init__(self):
        self.player_hands: List[BigTwoHand] = self.__create_player_hand()
        self.state: List[Tuple[int, List[Card]]] = []
        self.past_actions: List[Tuple[int, List[Card]]] = []
        self.current_player = None
        self.player_last_played = None
        self.n = 4

        self.event_count = {}

        if self.current_player is None:
            for idx, player_hand in enumerate(self.player_hands):
                if player_hand.have_diamond_three():
                    self.current_player = idx

        self.observation_space = spaces.Tuple(
            (
                # num_card_per_player
                spaces.Box(low=0, high=13, shape=(3,)),
                # last_cards_played
                spaces.Box(low=-1, high=51, shape=(5,)),
                # your_hands
                spaces.Box(low=-1, high=51, shape=(13,)),
                # current_player_number
                spaces.Discrete(4),
                # last_player_played
                spaces.Discrete(5),
            )
        )

        self.action_space = spaces.MultiBinary(13)

    @staticmethod
    def combination_order():
        return {
            BigTwo.STRAIGHT: 1,
            BigTwo.FLUSH: 2,
            BigTwo.FULL_HOUSE: 3,
            BigTwo.FOUR_OF_A_KIND: 4,
            BigTwo.STRAIGHT_FLUSH: 5,
        }

    @staticmethod
    def number_of_players():
        return 4

    @staticmethod
    def is_valid_card_combination(cards: List[Card]) -> bool:
        if len(cards) == 1:
            return True

        if len(cards) > 5 or len(cards) == 3 or len(cards) == 4:
            return False

        if len(cards) == 2:
            return cards[0].is_same_rank(cards[1])

        """
        When there is a 5 cards, it needs to fulfill either 4 criteria
            - all the same suit
            - the rank are consecutive
            - 3 cards are the same rank and the other 2 is the same rank
            - 4 cards are the same rank 
        """
        # checking if it's flush
        is_flush = True
        card_rank_count = {cards[0].rank: 1}
        for x in range(1, 5):
            if not cards[x - 1].is_same_suit(cards[x]):
                is_flush = False
            card_rank_count[cards[x].rank] = card_rank_count.get(cards[x].rank, 0) + 1

        if is_flush:
            return True

        if len(card_rank_count) == 2:
            return True

        # There isn't a valid combination with 3 or 4 ranks in it.
        if len(card_rank_count) == 3 or len(card_rank_count) == 4:
            return False

        return BigTwo.is_straight(cards)

    @staticmethod
    def is_straight(cards: List[Card]) -> bool:
        if len(cards) != 5:
            return False

        rank_map = {}
        for c in cards:
            rank_map[c.rank] = rank_map.get(c.rank, 0) + 1

        if len(rank_map.keys()) != 5:
            return False

        # straight_rank_order starts from ace, two, ... to king
        sorted_rank = sorted(rank_map.keys(), key=lambda r: straight_rank_order[r])

        first_rank_order = straight_rank_order[sorted_rank[0]]
        last_rank_order = straight_rank_order[sorted_rank[4]]

        # generaly the first and last card of a straight is 4 rank aparts
        if first_rank_order + 4 == last_rank_order:
            return True

        """
        special case:
        ten jack queen king ace
        """

        if sorted_rank[0] == Rank.ace and sorted_rank[1] == Rank.ten:
            return True

        return False

    @staticmethod
    def is_bigger(cards: List[Card], current_combination: List[Card]) -> bool:
        if len(cards) is not len(current_combination):
            return False

        if len(cards) not in [1, 2, 5]:
            raise ValueError("Number of cards is incorrect")

        if len(cards) == 1 or len(cards) == 2:
            rank_one = rank_order[cards[0].rank]
            rank_two = rank_order[current_combination[0].rank]

            if rank_one == rank_two:
                cards = sorted(cards, key=lambda c: suit_order[c.suit], reverse=True)
                current_combination = sorted(
                    current_combination, key=lambda c: suit_order[c.suit], reverse=True
                )
                suit_one = suit_order[cards[0].suit]
                suit_two = suit_order[current_combination[0].suit]
                return suit_one > suit_two

            return rank_one > rank_two

        combination_one = BigTwo.get_five_card_type(cards)
        combination_two = BigTwo.get_five_card_type(current_combination)

        combination_order = BigTwo.combination_order()
        if combination_one != combination_two:
            return (
                combination_order[combination_one] > combination_order[combination_two]
            )

        if combination_one is BigTwo.STRAIGHT_FLUSH or combination_one is BigTwo.FLUSH:
            suit_one = suit_order[cards[0].suit]
            suit_two = suit_order[current_combination[0].suit]

            if suit_one is not suit_two:
                return suit_one > suit_two

        # sort in Descending order to get the biggiest card to be first
        cards = sorted(cards, key=lambda c: rank_order[c.rank], reverse=True)
        current_combination = sorted(
            current_combination, key=lambda c: rank_order[c.rank], reverse=True
        )

        if (
            combination_one is BigTwo.STRAIGHT_FLUSH
            or combination_one is BigTwo.FLUSH
            or combination_one is BigTwo.STRAIGHT
        ):
            # straight rank
            # 3-4-5-6-7 < ... < 10-J-Q-K-A < 2-3-4-5-6 (Suit of 2 is tiebreaker) < A-2-3-4-5
            card_one_rank = cards[0].rank
            card_two_rank = current_combination[0].rank

            rank_one = rank_order[card_one_rank]
            rank_two = rank_order[card_two_rank]
            if rank_one == rank_two:
                # special case for A 2 3 4 4 and 2 3 4 5 6
                if cards[1].rank is not current_combination[1].rank:
                    return (
                        rank_order[cards[1].rank]
                        > rank_order[current_combination[1].rank]
                    )
                return (
                    suit_order[cards[0].suit] > suit_order[current_combination[0].suit]
                )
            return rank_one > rank_two

        # When it is Full House or Four of a kind
        rank_one_list = [card.rank for card in cards]
        rank_two_list = [card.rank for card in current_combination]

        if len(Counter(rank_one_list).values()) != 2:
            raise ValueError(
                "Expect only two different rank ie Full House or Four of a kind"
            )

        # most_common return a list of tuples
        common_one_rank = Counter(rank_one_list).most_common(1)[0][0]
        common_two_rank = Counter(rank_two_list).most_common(1)[0][0]
        return rank_order[common_one_rank] > rank_order[common_two_rank]

    @staticmethod
    def get_five_card_type(cards: List[Card]):
        if len(cards) != 5:
            raise ValueError("Need 5 cards to determine what type")

        is_same_suit = True
        card_rank_count = {}
        current_suit = cards[0].suit

        for card in cards:
            if card.suit is not current_suit:
                is_same_suit = False
            card_rank_count[card.rank] = card_rank_count.get(card.rank, 0) + 1

        if len(card_rank_count) == 2:
            rank = next(iter(card_rank_count))

            count = card_rank_count[rank]
            if count == 1 or count == 4:
                return BigTwo.FOUR_OF_A_KIND

            return BigTwo.FULL_HOUSE

        is_straight = BigTwo.is_straight(cards)
        if is_straight and is_same_suit:
            return BigTwo.STRAIGHT_FLUSH

        if is_same_suit:
            return BigTwo.FLUSH

        if is_straight:
            return BigTwo.STRAIGHT

        raise ValueError("invalid card combinations")

    def step(self, raw_action: List[int]) -> Tuple[BigTwoObservation, int, bool]:
        # check if raw_action is valid
        if len(raw_action) != 13:
            self.__increment_event("invalid raw action")
            return (
                self._current_observation(self.current_player),
                BigTwo.INVALID_MOVE_REWARD,
                False,
            )

        player_hand = self.player_hands[self.current_player]
        action = []
        for idx, value in enumerate(raw_action):
            if value == 1:
                if idx >= len(player_hand):
                    return (
                        self._current_observation(self.current_player),
                        BigTwo.INVALID_MOVE_REWARD,
                        False,
                    )
                action.append(player_hand[idx])

        self.past_actions.append((self.current_player, action))

        if len(action) == 0:
            # can't skip on the first turn of the game or when everyone else skipped already
            if self.is_first_turn() or self.player_last_played == self.current_player:
                event = (
                    "can't skip first turn"
                    if self.is_first_turn()
                    else "everyone else skipped"
                )
                self.__increment_event(event)
                return (
                    self._current_observation(self.current_player),
                    BigTwo.INVALID_MOVE_REWARD,
                    False,
                )

            # skipping
            previous_player = self.current_player
            self.current_player += 1
            if self.current_player > self.number_of_players() - 1:
                self.current_player = 0

            # 0 rewards for skipping
            return self._current_observation(previous_player), 0, False

        if not BigTwo.is_valid_card_combination(action):
            self.__increment_event("invalid combination")
            return (
                self._current_observation(self.current_player),
                BigTwo.INVALID_MOVE_REWARD,
                False,
            )

        if self.is_first_turn() or self.player_last_played == self.current_player:
            if self.is_first_turn() and not have_diamond_three(action):
                # always need to play diamond three first
                self.__increment_event("play diamond three")
                return (
                    self._current_observation(self.current_player),
                    BigTwo.INVALID_MOVE_REWARD,
                    False,
                )
            return self._apply_action(action)

        # there are cards played already
        if not BigTwo.is_bigger(action, self.get_current_combination()):
            # the cards that the player is played is not bigger than the existing combination so it's their turn again
            if len(action) == len(self.get_current_combination()):
                self.__increment_event("smaller combinations")
            else:
                self.__increment_event("unequal number of cards")
            return (
                self._current_observation(self.current_player),
                BigTwo.INVALID_MOVE_REWARD,
                False,
            )

        return self._apply_action(action)

    def __increment_event(self, event_name: str):
        self.event_count[event_name] = self.event_count.get(event_name, 0) + 1

    def reset(self) -> BigTwoObservation:
        self.player_hands = self.__create_player_hand()
        self.state = []
        self.past_actions = []
        self.event_count = {}

        self.current_player = None
        self.player_last_played = None

        if self.current_player is None:
            for idx, player_hand in enumerate(self.player_hands):
                if player_hand.have_diamond_three():
                    self.current_player = idx

        return self._current_observation(self._get_current_player())

    def is_rank_gt(self, rank_one: Rank, rank_two: Rank) -> bool:
        return rank_order[rank_one] > rank_order[rank_two]

    def is_suit_gt(self, suit_one: Suit, suit_two: Suit) -> bool:
        return suit_order[suit_one] > suit_order[suit_two]

    def have_playable_cards(self, target: List[Card], player_hand: BigTwoHand) -> bool:
        if len(target) == 0:
            return player_hand.have_diamond_three()

        if len(target) == 1:
            return self.__have_playable_single_card(target[0], player_hand)

        if len(target) == 2:
            return self.__have_playable_double_card(target, player_hand)

        return self.__have_playable_combination(target, player_hand)

    def __have_playable_single_card(self, card: Card, player_hand: BigTwoHand) -> bool:
        target_rank = card.rank
        target_suit = card.suit
        for card in player_hand:
            if card.rank == target_rank and self.is_suit_gt(card.suit, target_suit):
                return True

            if self.is_rank_gt(card.rank, target_rank):
                return True

        return False

    def __have_playable_double_card(
        self, pairs: List[Card], player_hand: BigTwoHand
    ) -> bool:
        rank_freq = {}
        rank_cards = {}
        target_rank = pairs[0].rank
        for card in player_hand:
            rank_freq[card.rank] = rank_freq.get(card.rank, 0) + 1
            card_list = rank_cards.get(card.rank, [])
            card_list.append(card)
            rank_cards[card.rank] = card_list

        if target_rank not in rank_freq:
            return False

        if rank_freq[target_rank] != 2:
            return False

        # need to figure out if the suit is higher
        max_target_suit = (
            pairs[0].suit
            if self.is_suit_gt(pairs[0].suit, pairs[1].suit)
            else pairs[1].suit
        )

        for card in rank_cards[target_rank]:
            if self.is_suit_gt(card.suit, max_target_suit):
                return True

        return False

    def __have_playable_combination(
        self, last_played_cards: List[Card], player_hand: BigTwoHand
    ) -> bool:
        if len(player_hand) < 5:
            return False

        current_c_type = BigTwo.get_five_card_type(last_played_cards)
        curr_c_type_rank = BigTwo.combination_order()[current_c_type]

        greater_c_type = [
            c_type
            for c_type, rank in BigTwo.combination_order().items()
            if rank > curr_c_type_rank
        ]

        for t in greater_c_type:
            if t in player_hand.combinations:
                return True

        # doesn't have the same combination in the available hand so return false because other possible combination
        # is smaller.
        if current_c_type not in player_hand.combinations:
            return False

        player_combinations = player_hand.combinations[current_c_type]
        for combinations in player_combinations:
            if BigTwo.is_bigger(list(combinations), last_played_cards):
                return True

        return False

    def is_first_turn(self) -> bool:
        return len(self.state) == 0

    def get_hands_played(self):
        return len(self.state)

    def get_starting_player_number(self) -> int:
        if len(self.state) == 0:
            return BigTwo.UNKNOWN_PLAYER

        player_number, _ = self.state[0]
        return player_number

    def get_current_combination(self) -> List[Card]:
        if len(self.state) == 0:
            return []

        return self.state[-1][1]

    def get_current_player_obs(self) -> BigTwoObservation:
        return self._current_observation(self._get_current_player())

    def get_player_obs(self, player_number: int):
        return self._current_observation(player_number)

    def display_all_player_hands(self) -> None:
        for idx, hand in enumerate(self.player_hands):
            print(idx, " ".join(str(x) for x in hand))

    def _get_current_player(self) -> int:
        if self.current_player is None:
            for idx, player_hand in enumerate(self.player_hands):
                if player_hand.have_diamond_three():
                    self.current_player = idx
                    return idx

            raise ValueError("One player should have Diamond three in their hand")

        return self.current_player

    def _current_observation(self, player_number: int) -> BigTwoObservation:
        num_card_per_player = []
        index = player_number
        while True:
            index += 1
            if index >= len(self.player_hands):
                index = 0

            num_card_per_player.append(len(self.player_hands[index]))

            if len(num_card_per_player) == len(self.player_hands) - 1:
                break

        last_cards_played = []
        if len(self.state) > 0:
            last_cards_played = self.get_current_combination()

        return BigTwoObservation(
            num_card_per_player,
            last_cards_played,
            self.player_hands[player_number],
            player_number,
            self.player_last_played
            if self.player_last_played is not None
            else self.UNKNOWN_PLAYER,
        )

    def _apply_action(self, action) -> Tuple[BigTwoObservation, int, bool]:
        self.state.append((self.current_player, action))
        self.player_hands[self.current_player].remove_cards(action)

        previous_player = self.current_player
        self.current_player += 1
        if self.current_player > self.number_of_players() - 1:
            self.current_player = 0
        self.player_last_played = previous_player

        game_finished = len(self.player_hands[previous_player]) == 0
        reward = 1000 if game_finished else 0
        return self._current_observation(previous_player), reward, game_finished

    def __create_player_hand(self) -> List[BigTwoHand]:
        hands = Deck().shuffle_and_split(self.number_of_players())
        return [BigTwoHand(hand) for hand in hands]
