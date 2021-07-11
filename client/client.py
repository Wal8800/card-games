import io
import itertools
import tkinter as tk
from typing import Dict

import cairosvg
from PIL import Image, ImageTk

from bigtwo.bigtwo import BigTwo, BigTwoHand
from playingcards.card import Suit, Rank

suit_image_mapping = {
    Suit.spades: "S",
    Suit.hearts: "H",
    Suit.clubs: "C",
    Suit.diamond: "D",
}


def display_opponent_cards(
    parent_window, x, y, cardback_img: ImageTk.PhotoImage, num_of_cards=13
):
    card_frame = tk.Frame(master=parent_window, relief=tk.RAISED, borderwidth=1)
    card_frame.grid(row=x, column=y)
    card_frame.columnconfigure(num_of_cards, weight=1)
    card_frame.rowconfigure(0, weight=1)

    for i in range(num_of_cards):
        cards = tk.Label(card_frame, image=cardback_img)
        cards.grid(row=0, column=i, columnspan=2)


def display_cards_played(
    parent_window, x: int, y: int, cardback_img: ImageTk.PhotoImage
):
    num_cards = 5
    card_frame = tk.Frame(master=parent_window, relief=tk.RAISED, borderwidth=1)
    card_frame.grid(row=x, column=y)
    card_frame.columnconfigure(num_cards, weight=1)
    card_frame.rowconfigure(0, weight=1)

    for i in range(num_cards):
        cards = tk.Label(card_frame, image=cardback_img)
        cards.grid(row=0, column=i)


def display_player_hand(
    parent_window,
    x: int,
    y: int,
    hand: BigTwoHand,
    card_images: Dict[str, ImageTk.PhotoImage],
):
    card_frame = tk.Frame(master=parent_window, relief=tk.RAISED, borderwidth=1)
    card_frame.grid(row=x, column=y)
    card_frame.columnconfigure(len(hand), weight=1)
    card_frame.rowconfigure(0, weight=1)

    for i, card in enumerate(hand):
        suit_str = suit_image_mapping[card.suit]
        key = card.rank.value + suit_str

        cards = tk.Label(card_frame, image=card_images[key])
        cards.grid(row=0, column=i)


def client():
    main = tk.Tk()

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
    card_back = ImageTk.PhotoImage(image)

    game = BigTwo()

    main.columnconfigure([0, 1, 2], weight=1, minsize=250)
    main.rowconfigure([0, 1, 2], weight=1, minsize=250)

    display_opponent_cards(main, 0, 1, card_back)
    display_opponent_cards(main, 1, 0, card_back)
    display_opponent_cards(main, 1, 2, card_back)

    display_cards_played(main, 1, 1, card_back)

    display_player_hand(main, 2, 1, game.player_hands[0], card_images)

    main.resizable(False, False)
    main.mainloop()


if __name__ == "__main__":
    client()
