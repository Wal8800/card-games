import unittest

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from bigtwo.bigtwo import BigTwoHand, BigTwoObservation, BigTwo
from gamerunner.ppo_bot import (
    create_action_cat_mapping,
    generate_action_mask,
    EmbeddedInputBot,
    SimplePPOBot,
    RandomPPOBot,
    cards_to_ohe,
    obs_to_ohe,
    PastCardsPlayedBot,
)
from playingcards.card import Card, Suit, Rank


class TestPPOBot(unittest.TestCase):
    def test_generate_action_mask_single(self):
        action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()

        cards = [Card(Suit.hearts, Rank.ace)]
        hand = BigTwoHand(cards)
        obs = BigTwoObservation([], [Card(Suit.hearts, Rank.six)], hand, 1, 3)

        mask = generate_action_mask(idx_cat_mapping, obs)

        self.assertEqual(mask[0], True)
        self.assertEqual(mask[1], True)
        self.assertEqual(mask[2], False)

    def test_generate_action_mask_single_smaller(self):
        action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()

        cards = [Card(Suit.hearts, Rank.three)]
        hand = BigTwoHand(cards)
        obs = BigTwoObservation([], [Card(Suit.hearts, Rank.six)], hand, 1, 3)

        mask = generate_action_mask(idx_cat_mapping, obs)

        self.assertEqual(mask[0], True)
        self.assertEqual(mask[1], False)
        self.assertEqual(mask[2], False)

    def test_generate_action_mask_double(self):
        action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()

        cards = [Card(Suit.hearts, Rank.ace), Card(Suit.spades, Rank.ace)]
        played = [Card(Suit.clubs, Rank.ten), Card(Suit.hearts, Rank.ten)]
        hand = BigTwoHand(cards)
        obs = BigTwoObservation([], played, hand, 1, 3)

        mask = generate_action_mask(idx_cat_mapping, obs)

        # can skip
        self.assertTrue(mask[0])

        # can't play single card
        self.assertFalse(mask[1])
        self.assertFalse(mask[2])

        # can play double cards
        self.assertTrue(mask[14])

        # other pairs is invalid
        self.assertFalse(mask[20])

    def test_generate_action_mask_double_smaller(self):
        action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()

        cards = [Card(Suit.hearts, Rank.seven), Card(Suit.spades, Rank.seven)]
        played = [Card(Suit.clubs, Rank.ten), Card(Suit.hearts, Rank.ten)]
        hand = BigTwoHand(cards)
        obs = BigTwoObservation([], played, hand, 1, 3)

        mask = generate_action_mask(idx_cat_mapping, obs)

        # can skip
        self.assertTrue(mask[0])

        # can play double cards
        self.assertFalse(mask[14])

    def test_generate_action_mask_comb(self):
        action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()

        cards = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.hearts, Rank.two),
            Card(Suit.hearts, Rank.four),
            Card(Suit.hearts, Rank.seven),
            Card(Suit.hearts, Rank.ten),
        ]

        played = [
            Card(Suit.spades, Rank.six),
            Card(Suit.clubs, Rank.seven),
            Card(Suit.diamond, Rank.eight),
            Card(Suit.spades, Rank.nine),
            Card(Suit.clubs, Rank.ten),
        ]

        hand = BigTwoHand(cards)
        obs = BigTwoObservation([], played, hand, 1, 3)

        mask = generate_action_mask(idx_cat_mapping, obs)

        # skip is valid
        self.assertTrue(mask[0])

        # single card 1 to 5 is invalid
        self.assertFalse(mask[1])
        self.assertFalse(mask[2])
        self.assertFalse(mask[3])
        self.assertFalse(mask[4])
        self.assertFalse(mask[5])

        self.assertTrue(mask[92])

        self.assertFalse(mask[100])

    def test_generate_action_mask_comb_smaller(self):
        action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()

        cards = [
            Card(Suit.clubs, Rank.six),
            Card(Suit.spades, Rank.seven),
            Card(Suit.hearts, Rank.eight),
            Card(Suit.diamond, Rank.nine),
            Card(Suit.diamond, Rank.ten),
        ]

        played = [
            Card(Suit.spades, Rank.six),
            Card(Suit.clubs, Rank.seven),
            Card(Suit.diamond, Rank.eight),
            Card(Suit.spades, Rank.nine),
            Card(Suit.clubs, Rank.ten),
        ]

        hand = BigTwoHand(cards)
        obs = BigTwoObservation([], played, hand, 1, 3)

        mask = generate_action_mask(idx_cat_mapping, obs)

        # skip is valid
        self.assertTrue(mask[0])

        # single card 1 to 5 is invalid
        self.assertFalse(mask[1])
        self.assertFalse(mask[2])
        self.assertFalse(mask[3])
        self.assertFalse(mask[4])
        self.assertFalse(mask[5])

        self.assertFalse(mask[92])

        self.assertFalse(mask[100])

    def test_generate_action_mask_no_skip(self):
        action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()

        cards = [Card(Suit.hearts, Rank.ace)]
        hand = BigTwoHand(cards)
        obs = BigTwoObservation([], [], hand, 1, 1)

        mask = generate_action_mask(idx_cat_mapping, obs)

        self.assertFalse(mask[0])

    def test_generate_action_mask_first_turn(self):
        action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()

        cards = [
            Card(Suit.diamond, Rank.three),
            Card(Suit.spades, Rank.three),
            Card(Suit.hearts, Rank.four),
            Card(Suit.spades, Rank.four),
            Card(Suit.clubs, Rank.four),
        ]
        hand = BigTwoHand(cards)
        obs = BigTwoObservation([], [], hand, 1, BigTwo.UNKNOWN_PLAYER)

        mask = generate_action_mask(idx_cat_mapping, obs)

        self.assertFalse(mask[0])

        # can play diamond three
        self.assertTrue(mask[1])

        # rest is invalid
        self.assertFalse(mask[2])
        self.assertFalse(mask[3])
        self.assertFalse(mask[4])
        self.assertFalse(mask[5])

        # can play double three
        self.assertTrue(mask[14])

        # other pairs is invalid
        self.assertFalse(mask[20])

        # can play combination including diamond three
        self.assertTrue(mask[92])

    def test_embedded_bot_action(self):
        env = BigTwo()

        init_obs = env.reset()
        bot = EmbeddedInputBot(init_obs)
        bot.action(init_obs)

    def test_simple_bot_action(self):
        env = BigTwo()

        init_obs = env.reset()
        bot = SimplePPOBot(init_obs)
        bot.action(init_obs)

    def test_random_bot_action(self):
        env = BigTwo()

        init_obs = env.reset()
        bot = RandomPPOBot()
        bot.action(init_obs)

    def test_sequence_input_bot_action(self):
        env = BigTwo()

        init_obs = env.reset()
        bot = PastCardsPlayedBot(init_obs)
        bot.action(init_obs)

    def test_set_weight_and_tf_function(self):
        inp = layers.Input(shape=(1,))
        x = layers.Dense(4, activation="relu")(inp)
        output = layers.Dense(1)(x)

        test_model = keras.Model(inputs=inp, outputs=output)

        weights = test_model.get_weights()

        @tf.function
        def test_action(model, v):
            return model(v)

        first_result = test_action(test_model, np.array([1])).numpy().flatten()

        weights[0] = np.array([[100.0, 200.0, 300.0, 400.0]], dtype=np.float32)
        test_model.set_weights(weights)

        second_result = test_action(test_model, np.array([1])).numpy().flatten()

        self.assertNotEqual(first_result, second_result)

    def test_obs_to_ohe(self):
        cards = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.clubs, Rank.ten),
            Card(Suit.diamond, Rank.two),
            Card(Suit.spades, Rank.seven),
            Card(Suit.hearts, Rank.king),
        ]

        played = []

        hand = BigTwoHand(cards)
        obs = BigTwoObservation([13, 13, 13], played, hand, 1, BigTwo.UNKNOWN_PLAYER)

        actual = obs_to_ohe(obs)

        # fmt: off
        expected = [
            # num_card_per_player
            13, 13, 13,

            # first turn
            1,

            # current player_number
            0, 1, 0, 0,

            # last player number
            0, 0, 0, 0,

            # cards length
            0,

            # last played cards
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

            # your hand
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
        # fmt: on

        np.testing.assert_allclose(actual, np.array(expected), rtol=1e-5, atol=0)

    def test_cards_to_ohe_full(self):
        cards = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.clubs, Rank.ten),
            Card(Suit.diamond, Rank.two),
            Card(Suit.spades, Rank.seven),
            Card(Suit.hearts, Rank.king),
        ]

        actual = cards_to_ohe(cards, 5)

        # fmt: off
        expected = [
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
        # fmt: on

        self.assertEqual(actual, expected)
