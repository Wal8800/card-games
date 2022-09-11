import itertools
from enum import Enum
from random import shuffle
from typing import List


class Suit(Enum):
    spades = "♠"
    hearts = "♥"
    clubs = "♣"
    diamond = "♦"


class Rank(Enum):
    ace = "A"
    king = "K"
    queen = "Q"
    jack = "J"
    ten = "10"
    nine = "9"
    eight = "8"
    seven = "7"
    six = "6"
    five = "5"
    four = "4"
    three = "3"
    two = "2"


class Card:
    SUIT_NUMBER = {Suit.spades: 0, Suit.hearts: 1, Suit.clubs: 2, Suit.diamond: 3}

    RANK_NUMBER = {
        Rank.ace: 0,
        Rank.two: 1,
        Rank.three: 2,
        Rank.four: 3,
        Rank.five: 4,
        Rank.six: 5,
        Rank.seven: 6,
        Rank.eight: 7,
        Rank.nine: 8,
        Rank.ten: 9,
        Rank.jack: 10,
        Rank.queen: 11,
        Rank.king: 12,
    }

    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank

    def __repr__(self):
        return self.suit.value + "" + self.rank.value

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.suit == other.suit and self.rank == other.rank
        return False

    def __hash__(self):
        return hash(self.__repr__())

    @staticmethod
    def from_number(number: int):
        if not 0 <= number <= 51:
            raise ValueError("Invalid card number, must be from 0 to 51")

        rank = number % 13
        suit = number // 13

        inverse_suit_number = {v: k for k, v in Card.SUIT_NUMBER.items()}
        inverse_rank_number = {v: k for k, v in Card.RANK_NUMBER.items()}

        return Card(inverse_suit_number[suit], inverse_rank_number[rank])

    @staticmethod
    def display_cards_string(cards):
        filtered_list = [x for x in cards if x > -1]
        return " ".join(str(Card.from_number(x)) for x in filtered_list)

    def is_same_suit(self, card) -> bool:
        return self.suit == card.suit

    def is_same_rank(self, card) -> bool:
        return self.rank == card.rank

    def to_number(self):
        """
        Used to generated observation
        :return: int
        """
        return (Card.SUIT_NUMBER[self.suit] * 13) + Card.RANK_NUMBER[self.rank]


class Deck:
    def __init__(self):
        self.cards = []

        for element in itertools.product(
            Suit.__members__.items(), Rank.__members__.items()
        ):
            suit = element[0][1]
            rank = element[1][1]

            card = Card(suit, rank)

            self.cards.append(card)

    def display_all_cards(self):
        for card in self.cards:
            print(card)

    def shuffle_deck(self):
        shuffle(self.cards)

    def shuffle_and_split(self, n: int) -> List[List[Card]]:
        self.shuffle_deck()
        return [self.cards[i::n] for i in range(n)]
