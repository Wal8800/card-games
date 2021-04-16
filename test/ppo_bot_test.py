import unittest

from bigtwo.bigtwo import BigTwoHand, BigTwoObservation, BigTwo
from gamerunner.ppo_bot import (
    create_action_cat_mapping,
    generate_action_mask,
    EmbeddedInputBot,
    SimplePPOBot,
)
from playingcards.card import Card, Suit, Rank


class TestPPOBot(unittest.TestCase):
    def test_generate_action_mask_single(self):
        action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()

        cards = [Card(Suit.hearts, Rank.ace)]
        hand = BigTwoHand(cards)
        obs = BigTwoObservation([], [Card(Suit.hearts, Rank.six)], hand, 1, 3)

        mask = generate_action_mask(action_cat_mapping, idx_cat_mapping, obs)

        self.assertEqual(mask[0], True)
        self.assertEqual(mask[1], True)
        self.assertEqual(mask[2], False)

    def test_generate_action_mask_double(self):
        action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()

        cards = [Card(Suit.hearts, Rank.ace), Card(Suit.spades, Rank.ace)]
        played = [Card(Suit.clubs, Rank.ten), Card(Suit.hearts, Rank.ten)]
        hand = BigTwoHand(cards)
        obs = BigTwoObservation([], played, hand, 1, 3)

        mask = generate_action_mask(action_cat_mapping, idx_cat_mapping, obs)

        # can skip
        self.assertTrue(mask[0])

        # can't play single card
        self.assertFalse(mask[1])
        self.assertFalse(mask[2])

        # can play double cards
        self.assertEqual(mask[14], True)

        # other pairs is invalid
        self.assertEqual(mask[20], False)

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

        mask = generate_action_mask(action_cat_mapping, idx_cat_mapping, obs)

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

    def test_generate_action_mask_no_skip(self):
        action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()

        cards = [Card(Suit.hearts, Rank.ace)]
        hand = BigTwoHand(cards)
        obs = BigTwoObservation([], [], hand, 1, 1)

        mask = generate_action_mask(action_cat_mapping, idx_cat_mapping, obs)

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

        mask = generate_action_mask(action_cat_mapping, idx_cat_mapping, obs)

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
