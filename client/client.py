import io
import itertools
import tkinter as tk
from typing import Dict

import cairosvg
from PIL import Image, ImageTk

from bigtwo.bigtwo import BigTwo, BigTwoHand
from gamerunner.ppo_bot import SavedPPOBot
from gamerunner.ppo_runner import SinglePlayerWrapper
from playingcards.card import Suit, Rank

suit_image_mapping = {
    Suit.spades: "S",
    Suit.hearts: "H",
    Suit.clubs: "C",
    Suit.diamond: "D",
}


class BigTwoClient:
    CARD_BACK = "BACK"

    def __init__(self):
        game = BigTwo()
        init_obs = game.get_player_obs(0)

        # 3 bots
        opponent_bots = [
            SavedPPOBot(
                dir_path="../gamerunner/save/2021_07_10_10_21_42",
                observation=init_obs,
            )
            for _ in range(3)
        ]

        self.wrapped_env = SinglePlayerWrapper(
            env=game, opponent_bots=opponent_bots, single_player_number=0
        )

        obs = self.wrapped_env.reset_and_start()

        self.main = tk.Tk()
        self.main.columnconfigure([0, 1, 2], weight=1, minsize=250)
        self.main.rowconfigure([0, 1, 2], weight=1, minsize=250)

        self.card_images = self._load_card_images()

        self._add_opponents_cards(0, 1)
        self._add_opponents_cards(1, 0)
        self._add_opponents_cards(1, 2)

        self._add_cards_played(1, 1)
        self._add_player_hand(2, 1, obs.your_hands)

        self._add_control_frame(2, 2)

        self.main.resizable(False, False)

    def _load_card_images(self) -> Dict[str, ImageTk.PhotoImage]:
        card_images = {}
        for r, s in itertools.product(Rank, Suit):
            suit_str = suit_image_mapping[s]
            key = r.value + suit_str
            path = f"cards/{key}.svg"
            data = cairosvg.svg2png(url=path, output_width=75)
            img = Image.open(io.BytesIO(data))
            card_images[key] = ImageTk.PhotoImage(img)

        image_data = cairosvg.svg2png(url="cards/BLUE_BACK.svg", output_width=75)
        image = Image.open(io.BytesIO(image_data))
        card_images[self.CARD_BACK] = ImageTk.PhotoImage(image)

        return card_images

    def _add_opponents_cards(self, x, y, num_of_cards=13):
        card_frame = tk.Frame(master=self.main, relief=tk.RAISED, borderwidth=1)
        card_frame.grid(row=x, column=y)
        card_frame.columnconfigure(num_of_cards, weight=1)
        card_frame.rowconfigure(0, weight=1)

        for i in range(num_of_cards):
            cards = tk.Label(card_frame, image=self.card_images[self.CARD_BACK])
            cards.grid(row=0, column=i, columnspan=2)

    def _add_control_frame(self, x: int, y: int):
        control_frame = tk.Frame(master=self.main, relief=tk.RAISED, borderwidth=1)
        control_frame.grid(row=x, column=y)
        control_frame.columnconfigure([0, 1], weight=1)
        control_frame.rowconfigure([0, 1], weight=1)

        play_button = tk.Button(
            master=control_frame,
            text="Play Cards",
            width=10,
            height=5,
        )
        play_button.grid(row=0, column=0)

        reset_button = tk.Button(
            master=control_frame,
            text="Reset",
            width=10,
            height=5,
        )
        reset_button.grid(row=0, column=1)

        sort_by_rank_button = tk.Button(
            master=control_frame,
            text="Sort By Rank",
            width=10,
            height=5,
        )
        sort_by_rank_button.grid(row=1, column=0)

        sort_by_suit_button = tk.Button(
            master=control_frame,
            text="Sort By Suit",
            width=10,
            height=5,
        )
        sort_by_suit_button.grid(row=1, column=1)

    def _add_cards_played(self, x: int, y: int):
        num_cards = 5
        card_frame = tk.Frame(master=self.main, relief=tk.RAISED, borderwidth=1)
        card_frame.grid(row=x, column=y)
        card_frame.columnconfigure(num_cards, weight=1)
        card_frame.rowconfigure(0, weight=1)

        for i in range(num_cards):
            cards = tk.Label(card_frame, image=self.card_images[self.CARD_BACK])
            cards.grid(row=0, column=i)

    def _add_player_hand(
        self,
        x: int,
        y: int,
        hand: BigTwoHand,
    ):
        card_frame = tk.Frame(master=self.main, relief=tk.RAISED, borderwidth=1)
        card_frame.grid(row=x, column=y)
        card_frame.columnconfigure(len(hand), weight=1)
        card_frame.rowconfigure([0, 1], weight=1)

        for i, card in enumerate(hand):
            suit_str = suit_image_mapping[card.suit]
            key = card.rank.value + suit_str

            cards = tk.Label(card_frame, image=self.card_images[key])
            cards.grid(row=0, column=i)

            check = tk.Checkbutton(card_frame)
            check.grid(row=1, column=i)

    def run(self):
        self.main.mainloop()


def main():
    client = BigTwoClient()
    client.run()


if __name__ == "__main__":
    main()
