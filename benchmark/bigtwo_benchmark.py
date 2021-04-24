import time

from bigtwo.bigtwo import (
    find_combinations_from_cards,
    bf_find_combinations_from_cards,
    straight_rank_order,
)
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


def check__find_combinations_from_cards():
    deck = Deck()

    for _ in range(100):
        hands = deck.shuffle_and_split(4)
        for hand in hands:
            c1 = bf_find_combinations_from_cards(hand)

            c1 = {
                k: [sorted(c, key=lambda r: straight_rank_order[r.rank]) for c in v]
                for k, v in c1.items()
            }
            c2 = find_combinations_from_cards(hand)

            for c_type in c1:
                if c_type not in c2:
                    print(c1)
                    print(c2)
                    return

                if len(c2[c_type]) != len(c1[c_type]):
                    print(c1)
                    print(c2)
                    return


if __name__ == "__main__":
    check__find_combinations_from_cards()
