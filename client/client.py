import io
import itertools
import tkinter as tk
from typing import List

import cairosvg
from PIL import Image, ImageTk

from bigtwo.bigtwo import BigTwo, BigTwoHand, BigTwoObservation
from gamerunner.ppo_bot import SavedPPOBot, PPOAction
from gamerunner.ppo_runner import SinglePlayerWrapper
from playingcards.card import Suit, Rank, Card

suit_image_mapping = {
    Suit.spades: "S",
    Suit.hearts: "H",
    Suit.clubs: "C",
    Suit.diamond: "D",
}


class OpponentCardFrame:
    def __init__(self, parent, x, y, card_back, num_of_card):
        self.card_frame = tk.Frame(master=parent, relief=tk.RAISED, borderwidth=1)
        self.card_frame.grid(row=x, column=y)

        self.card_back_img = card_back

        self.update_cards(num_of_card)

    def update_cards(self, num_of_cards, clear=False):
        if clear:
            for widget in self.card_frame.winfo_children():
                widget.destroy()

        self.card_frame.columnconfigure(num_of_cards, weight=1)
        self.card_frame.rowconfigure([0, 1], weight=1)

        last_played_label = tk.Label(self.card_frame, text=f"{num_of_cards}")
        last_played_label.grid(row=1, column=num_of_cards // 2)

        columnspan = 2 if num_of_cards > 8 else 1
        for i in range(num_of_cards):
            cards = tk.Label(self.card_frame, image=self.card_back_img)
            cards.grid(row=0, column=i, columnspan=columnspan)


class CardImages:
    CARD_BACK = "BACK"

    def __init__(self):
        image_width = 70
        self.card_images = {}
        for r, s in itertools.product(Rank, Suit):
            suit_str = suit_image_mapping[s]
            key = r.value + suit_str
            path = f"cards/{key}.svg"
            data = cairosvg.svg2png(url=path, output_width=image_width)
            img = Image.open(io.BytesIO(data))
            self.card_images[key] = ImageTk.PhotoImage(img)

        image_data = cairosvg.svg2png(
            url="cards/BLUE_BACK.svg", output_width=image_width
        )
        image = Image.open(io.BytesIO(image_data))

        self.card_images[self.CARD_BACK] = ImageTk.PhotoImage(image)

    def get_card_image(self, card: Card) -> ImageTk.PhotoImage:
        suit_str = suit_image_mapping[card.suit]
        key = card.rank.value + suit_str
        return self.card_images[key]

    def get_card_back(self) -> ImageTk.PhotoImage:
        return self.card_images[self.CARD_BACK]


class PlayedCardFrame:
    def __init__(self, parent, x, y, obs: BigTwoObservation, card_images: CardImages):
        self.num_cards = 5
        self.card_frame = tk.Frame(master=parent, relief=tk.RAISED, borderwidth=1)
        self.card_frame.grid(row=x, column=y)

        self.card_frame.columnconfigure(self.num_cards, weight=1)
        self.card_frame.rowconfigure([0, 1], weight=1)

        self.card_images = card_images
        self.last_played_mapping = {0: "YOU", 1: "LEFT", 2: "TOP", 3: "RIGHT"}

        self.update_cards(obs)

    def update_cards(self, obs: BigTwoObservation, clear=False):
        if clear:
            for widget in self.card_frame.winfo_children():
                widget.destroy()

        label_txt = self.last_played_mapping.get(obs.last_player_played, "")

        last_played_label = tk.Label(self.card_frame, text=label_txt)
        last_played_label.grid(row=1, column=2)

        for i, card in enumerate(obs.last_cards_played):
            cards = tk.Label(
                self.card_frame, image=self.card_images.get_card_image(card)
            )
            cards.grid(row=0, column=i)

        for i in range(len(obs.last_cards_played), self.num_cards):
            cards = tk.Label(self.card_frame, image=self.card_images.get_card_back())
            cards.grid(row=0, column=i)


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

        self.card_images = CardImages()

        self.left_player_frame = OpponentCardFrame(
            self.main,
            1,
            0,
            self.card_images.get_card_back(),
            obs.num_card_per_player[0],
        )
        self.top_player_frame = OpponentCardFrame(
            self.main,
            0,
            1,
            self.card_images.get_card_back(),
            obs.num_card_per_player[1],
        )
        self.right_player_frame = OpponentCardFrame(
            self.main,
            1,
            2,
            self.card_images.get_card_back(),
            obs.num_card_per_player[2],
        )

        self.selected_idx: List[tk.IntVar] = []

        self.card_played_frame = PlayedCardFrame(self.main, 1, 1, obs, self.card_images)

        self.player_card_frame = tk.Frame(
            master=self.main, relief=tk.RAISED, borderwidth=1
        )
        self.player_card_frame.grid(row=2, column=1)

        self._update_play_hand(self.player_card_frame, obs.your_hands)

        self._add_control_frame(2, 2)

        self.main.geometry("1920x1080")
        self.main.resizable(width=0, height=0)

    def _update_game(self, obs: BigTwoObservation):
        self.left_player_frame.update_cards(obs.num_card_per_player[0], clear=True)
        self.top_player_frame.update_cards(obs.num_card_per_player[1], clear=True)
        self.right_player_frame.update_cards(obs.num_card_per_player[2], clear=True)

        self.card_played_frame.update_cards(obs, clear=True)

        self._update_play_hand(self.player_card_frame, obs.your_hands, clear=True)

    def _reset_game(self):
        obs = self.wrapped_env.reset_and_start()

        self._update_game(obs)

    def _play_selected_idx(self):
        raw_action = [1 if v.get() >= 0 else 0 for v in self.selected_idx]

        if len(raw_action) < 13:
            raw_action += (13 - len(raw_action)) * [0]

        action = PPOAction(
            transformed_obs=None,
            raw_action=raw_action,
            action_cat=None,
            action_mask=None,
            logp=None,
            cards=None,
        )

        obs, rewards, done = self.wrapped_env.step(action)

        self._update_game(obs)

    def _add_control_frame(self, x: int, y: int):
        control_frame = tk.Frame(master=self.main, relief=tk.RAISED, borderwidth=1)
        control_frame.grid(row=x, column=y)
        control_frame.columnconfigure([0, 1], weight=1)
        control_frame.rowconfigure([0, 1], weight=1)

        play_button = tk.Button(
            master=control_frame,
            command=self._play_selected_idx,
            text="Play Cards",
            width=10,
            height=5,
        )
        play_button.grid(row=0, column=0)

        reset_button = tk.Button(
            master=control_frame,
            command=self._reset_game,
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

    def _update_play_hand(self, frame, hand: BigTwoHand, clear=False):
        if clear:
            for widget in frame.winfo_children():
                widget.destroy()

        frame.columnconfigure(len(hand), weight=1)
        frame.rowconfigure([0, 1], weight=1)
        self.selected_idx = []
        for i, card in enumerate(hand):
            cards = tk.Label(frame, image=self.card_images.get_card_image(card))
            cards.grid(row=0, column=i)

            idx = tk.IntVar(value=-1)
            check = tk.Checkbutton(frame, variable=idx, onvalue=i, offvalue=-1)
            check.grid(row=1, column=i)

            self.selected_idx.append(idx)

    def run(self):
        self.main.mainloop()


def main():
    client = BigTwoClient()
    client.run()


if __name__ == "__main__":
    main()
