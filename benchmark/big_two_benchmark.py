import time

from bigtwo.bigtwo import find_combinations_from_cards, bf_find_combinations_from_cards
from playingcards.card import Deck


def bench_find_combinations_from_cards():
    st = time.time()
    deck = Deck()
    for _ in range(1000):
        hands = deck.shuffle_and_split(4)
        _ = bf_find_combinations_from_cards(hands[0])

    print("bf: ", time.time() - st)

    st = time.time()
    deck = Deck()
    for _ in range(1000):
        hands = deck.shuffle_and_split(4)
        _ = find_combinations_from_cards(hands[0])

    print("custom: ", time.time() - st)



if __name__ == "__main__":
    bench_find_combinations_from_cards()
