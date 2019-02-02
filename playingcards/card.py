from enum import Enum
from random import shuffle
import itertools


class Suit(Enum):
    spades = 1
    hearts = 2
    clubs = 3
    diamond = 4


class Rank(Enum):
    ace = 1
    king = 2
    queen = 3
    jack = 4
    ten = 5
    nine = 6
    eight = 7
    seven = 8
    six = 9
    five = 10
    four = 11
    three = 12
    two = 13


class Card:
    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return self.suit.name + " " + self.rank.name

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

