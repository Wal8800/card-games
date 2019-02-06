from enum import Enum
from random import shuffle
import itertools


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
    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return self.suit.value + "" + self.rank.value

    def is_same_suit(self, card):
        return self.suit == card.suit

    def is_same_rank(self, card):
        return self.rank == card.rank


class Deck:
    def __init__(self):
        self.cards = []

        for element in itertools.product(Suit.__members__.items(), Rank.__members__.items()):
            suit = element[0][1]
            rank = element[1][1]

            card = Card(suit, rank)

            self.cards.append(card)

    def display_all_cards(self):
        for card in self.cards:
            print(card)

    def shuffle_deck(self):
        shuffle(self.cards)

    def shuffle_and_split(self, n):
        self.shuffle_deck()
        return [self.cards[i::n] for i in range(n)]

