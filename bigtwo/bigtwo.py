from collections import Counter

from gym import spaces
from typing import List, Dict, Tuple

from playingcards.card import Card, Deck, Suit, Rank


def have_diamond_three(hand):
    for card in hand:
        if card.suit == Suit.diamond and card.rank == Rank.three:
            return True

    return False


class BigTwoObservation:
    def __init__(self, num_card_per_player: List[int], last_cards_played: List[Card], your_hands: List[Card],
                 current_player: int, last_player_played: int):
        self.num_card_per_player = num_card_per_player
        self.last_cards_played = last_cards_played
        self.your_hands = your_hands
        self.current_player = current_player
        self.last_player_played = last_player_played


class BigTwo:
    FULL_HOUSE = 1
    FOUR_OF_A_KIND = 2
    STRAIGHT = 3
    FLUSH = 4
    STRAIGHT_FLUSH = 5

    """
    TODO
     - can't skip once everyone else skipped
    """

    def __init__(self):
        self.player_hands = Deck().shuffle_and_split(self.number_of_players())
        self.state = []
        self.current_player = None
        self.player_last_played = None
        self.n = 4

        if self.current_player is None:
            for idx, player_hand in enumerate(self.player_hands):
                if have_diamond_three(player_hand):
                    self.current_player = idx

        self.observation_space = spaces.Tuple((
            # num_card_per_player
            spaces.Box(low=0, high=13, shape=(3,)),
            # last_cards_played
            spaces.Box(low=-1, high=51, shape=(5,)),
            # your_hands
            spaces.Box(low=-1, high=51, shape=(13,)),
            # current_player_number
            spaces.Discrete(4),
            # last_player_played
            spaces.Discrete(5)
        ))

        self.action_space = spaces.MultiBinary(13)

    @staticmethod
    def rank_order() -> Dict[Rank, int]:
        return {
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
            Rank.two: 13
        }

    @staticmethod
    def suit_order() -> Dict[Suit, int]:
        return {
            Suit.diamond: 1,
            Suit.clubs: 2,
            Suit.hearts: 3,
            Suit.spades: 4
        }

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
        card_rank_count = {
            cards[0].rank: 1
        }
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
        rank_order_map = BigTwo.rank_order()
        cards = sorted(cards, key=lambda c: rank_order_map[c.rank])

        """
        special case:
    
        3 4 5 ace two
        3 4 5 6 two
        """
        for x in range(1, 5):
            card_one = cards[x - 1]
            card_two = cards[x]

            card_one_order = rank_order_map[card_one.rank]
            card_two_order = rank_order_map[card_two.rank]

            # special case, can ignore
            if card_one.rank == Rank.five and card_two.rank == Rank.ace or \
                    card_one.rank == Rank.six and card_two.rank == Rank.two:
                continue

            if card_one_order + 1 != card_two_order:
                return False

        return True

    @staticmethod
    def is_bigger(cards: List[Card], current_combination: List[Card]) -> bool:
        if len(cards) is not len(current_combination):
            return False

        if len(cards) not in [1, 2, 5]:
            raise ValueError("Number of cards is incorrect")

        rank_order = BigTwo.rank_order()
        suit_order = BigTwo.suit_order()

        if len(cards) == 1 or len(cards) == 2:
            rank_one = rank_order[cards[0].rank]
            rank_two = rank_order[current_combination[0].rank]

            if rank_one == rank_two:
                cards = sorted(cards, key=lambda c: suit_order[c.suit], reverse=True)
                current_combination = sorted(current_combination, key=lambda c: suit_order[c.suit], reverse=True)
                suit_one = suit_order[cards[0].suit]
                suit_two = suit_order[current_combination[0].suit]
                return suit_one > suit_two

            return rank_one > rank_two

        combination_one = BigTwo.get_five_card_type(cards)
        combination_two = BigTwo.get_five_card_type(current_combination)

        combination_order = BigTwo.combination_order()
        if combination_one != combination_two:
            return combination_order[combination_one] > combination_order[combination_two]

        if combination_one is BigTwo.STRAIGHT_FLUSH or combination_one is BigTwo.FLUSH:
            suit_one = suit_order[cards[0].suit]
            suit_two = suit_order[current_combination[0].suit]

            if suit_one is not suit_two:
                return suit_one > suit_two

        # sort in Descending order to get the biggiest card to be first
        cards = sorted(cards, key=lambda c: rank_order[c.rank], reverse=True)
        current_combination = sorted(current_combination, key=lambda c: rank_order[c.rank], reverse=True)

        if combination_one is BigTwo.STRAIGHT_FLUSH or combination_one is BigTwo.FLUSH or combination_one is BigTwo.STRAIGHT:
            # straight rank
            # 3-4-5-6-7 < ... < 10-J-Q-K-A < 2-3-4-5-6 (Suit of 2 is tiebreaker) < A-2-3-4-5
            card_one_rank = cards[0].rank
            card_two_rank = current_combination[0].rank

            rank_one = rank_order[card_one_rank]
            rank_two = rank_order[card_two_rank]
            if rank_one == rank_two:
                # special case for A 2 3 4 4 and 2 3 4 5 6
                if cards[1].rank is not current_combination[1].rank:
                    return rank_order[cards[1].rank] > rank_order[current_combination[1].rank]
                return suit_order[cards[0].suit] > suit_order[current_combination[0].suit]
            return rank_one > rank_two

        # When it is Full House or Four of a kind
        rank_one_list = [card.rank for card in cards]
        rank_two_list = [card.rank for card in current_combination]

        if len(Counter(rank_one_list).values()) != 2:
            raise ValueError("Expect only two different rank ie Full House or Four of a kind")

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

    @staticmethod
    def remove_card_from_hand(cards, hand):
        for card in cards:
            for card_in_hand in hand:
                if card == card_in_hand:
                    hand.remove(card_in_hand)
                    break

        return hand

    def step(self, raw_action: List[int]) -> Tuple[BigTwoObservation, int, bool]:
        # check if raw_action is valid
        if len(raw_action) != 13:
            return self._current_observation(self.current_player), -1, False

        player_hand = self.player_hands[self.current_player]
        action = []
        for idx, value in enumerate(raw_action):
            if value == 1:
                if idx >= len(player_hand):
                    return self._current_observation(self.current_player), -1, False
                action.append(player_hand[idx])

        if len(action) == 0:
            # can't skip on the first turn of the game or when everyone else skipped already
            if self.is_first_turn() or self.player_last_played == self.current_player:
                return self._current_observation(self.current_player), -1, False

            # skipping
            previous_player = self.current_player
            self.current_player += 1
            if self.current_player > self.number_of_players() - 1:
                self.current_player = 0

            return self._current_observation(previous_player), 0, False

        if not BigTwo.is_valid_card_combination(action):
            return self._current_observation(self.current_player), -1, False

        if self.is_first_turn() or self.player_last_played == self.current_player:
            if self.is_first_turn() and not have_diamond_three(action):
                # always need to play diamond three first
                return self._current_observation(self.current_player), -1, False
            return self._apply_action(action)

        # there are cards played already
        current_combination = self.state[len(self.state) - 1]

        if not BigTwo.is_bigger(action, current_combination):
            # the cards that the player is played is not bigger than the existing combination so it's their turn again
            return self._current_observation(self.current_player), -1, False

        return self._apply_action(action)

    def reset(self) -> BigTwoObservation:
        self.player_hands = Deck().shuffle_and_split(self.number_of_players())
        self.state = []

        self.current_player = None
        self.player_last_played = None

        if self.current_player is None:
            for idx, player_hand in enumerate(self.player_hands):
                if have_diamond_three(player_hand):
                    self.current_player = idx

        return self._current_observation(self._get_current_player())

    def is_first_turn(self) -> bool:
        return len(self.state) == 0

    def get_hands_played(self):
        return len(self.state)

    def get_current_player_obs(self) -> BigTwoObservation:
        return self._current_observation(self._get_current_player())

    def get_player_obs(self, player_number: int):
        return self._current_observation(player_number)

    def display_all_player_hands(self) -> None:
        for idx, hand in enumerate(self.player_hands):
            print(idx, ' '.join(str(x) for x in hand))

    def _get_current_player(self) -> int:
        if self.current_player is None:
            for idx, player_hand in enumerate(self.player_hands):
                if have_diamond_three(player_hand):
                    self.current_player = idx
                    return idx

            raise ValueError("One player should have Diamond three in their hand")

        return self.current_player

    def _current_observation(self, player_number: int) -> BigTwoObservation:
        num_card_per_player = []
        while True:
            index = player_number + 1
            if index >= len(self.player_hands):
                index = 0

            num_card_per_player.append(len(self.player_hands[index]))

            if len(num_card_per_player) == len(self.player_hands) - 1:
                break

        last_cards_played = []
        if len(self.state) > 0:
            last_cards_played = self.state[-1]

        return BigTwoObservation(
            num_card_per_player,
            last_cards_played,
            self.player_hands[player_number],
            player_number,
            self.player_last_played if self.player_last_played is not None else 5
        )

    def _apply_action(self, action) -> Tuple[BigTwoObservation, int, bool]:
        self.state.append(action)

        self.player_hands[self.current_player] = BigTwo.remove_card_from_hand(
            action,
            self.player_hands[self.current_player]
        )

        previous_player = self.current_player
        self.current_player += 1
        if self.current_player > self.number_of_players() - 1:
            self.current_player = 0
        self.player_last_played = previous_player

        reward = 1 * len(action)
        game_finished = len(self.player_hands[previous_player]) == 0
        if game_finished:
            reward = 100
        return self._current_observation(previous_player), reward, game_finished
