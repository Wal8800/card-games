import copy
import itertools
import pickle
import time
from collections import Counter
from datetime import datetime
from multiprocessing import Process, Queue
from typing import List

from tqdm import trange

from bigtwo.bigtwo import (
    STRAIGHT_RANK_ORDER,
    BigTwo,
    BigTwoHand,
    bf_find_combinations_from_cards,
    find_combinations_from_cards,
)
from gamerunner.ppo_runner import build_bot
from playingcards.card import Card, Deck


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


def check_find_combinations_from_cards():
    deck = Deck()

    for _ in range(100):
        hands = deck.shuffle_and_split(4)
        for hand in hands:
            c1 = bf_find_combinations_from_cards(hand)

            c1 = {
                k: [sorted(c, key=lambda r: STRAIGHT_RANK_ORDER[r.rank]) for c in v]
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


def gen_benchmark_hands_list():
    deck = Deck()

    benchmark_hands_list = [deck.shuffle_and_split(4) for _ in range(1000)]
    current_timestamp = datetime.now().strftime("%Y_%m_%d")
    pickle.dump(
        benchmark_hands_list,
        open(f"./benchmark_hands_list_{current_timestamp}.pickle", "wb"),
    )


def sample_worker(input_queue: Queue, output: Queue):
    bots = {
        "player_one": build_bot(
            dir_path="../gamerunner/experiments/2021_07_13_21_31_32/bot_save/2021_07_13_21_31_32_10000",
        ),
        "player_two": build_bot(
            dir_path="../gamerunner/experiments/2021_07_13_21_31_32/bot_save/2021_07_13_21_31_32_10000",
        ),
        "player_three": build_bot(
            dir_path="../gamerunner/experiments/2021_07_13_21_31_32/bot_save/2021_07_13_21_31_32_10000",
        ),
        "player_four": build_bot(
            dir_path="../gamerunner/experiments/2021_07_13_21_31_32/bot_save/2021_07_13_21_31_32_10000",
        ),
    }

    try:
        for hands in iter(input_queue.get, "STOP"):
            for player_order in itertools.permutations(bots.keys()):
                copies = copy.deepcopy(hands)
                bigtwo_hands = [BigTwoHand(hand) for hand in copies]
                bigtwo = BigTwo(bigtwo_hands)

                while True:
                    obs = bigtwo.get_current_player_obs()

                    player_key = player_order[obs.current_player]
                    bot = bots[player_key]

                    action = bot.action(obs)
                    _, done = bigtwo.step(action.raw)

                    if done:
                        output.put(player_key)
                        break
    except RuntimeError:
        output.put("ERROR")


def run_benchmark_hands_list():
    NUMBER_OF_PROCESSES = 10

    task_queue = Queue()
    output_queue = Queue()

    benchmark_result = []

    with open("./benchmark_hands_list_2021_08_07.pickle", "rb") as pickle_file:
        fixture: List[List[List[Card]]] = pickle.load(pickle_file)

    try:
        # Start worker processes
        for i in range(NUMBER_OF_PROCESSES):
            Process(target=sample_worker, args=(task_queue, output_queue)).start()

        for hands in fixture:
            task_queue.put(hands)

        task_queue.qsize()
        for _ in trange(24 * len(fixture)):
            result = output_queue.get()

            if result == "ERROR":
                break

            benchmark_result.append(result)
    finally:
        print("stopping workers")
        # Tell child processes to stop
        for i in range(NUMBER_OF_PROCESSES):
            task_queue.put("STOP")

    c = Counter(benchmark_result)
    print(c.most_common())


if __name__ == "__main__":
    run_benchmark_hands_list()
