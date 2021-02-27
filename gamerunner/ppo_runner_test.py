import unittest

from bigtwo.bigtwo import BigTwoHand
from gamerunner.ppo_runner import create_action_cat_mapping, generate_action_mask
from playingcards.card import Card, Suit, Rank


class TestPPORunner(unittest.TestCase):
    def test_generate_action_mask_single(self):
        action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()

        cards = [
            Card(Suit.hearts, Rank.ace)
        ]

        hand = BigTwoHand(cards)

        mask = generate_action_mask(action_cat_mapping, idx_cat_mapping, hand)

        self.assertEqual(mask[0], True)
        self.assertEqual(mask[1], True)
        self.assertEqual(mask[2], False)

    def test_generate_action_mask_double(self):
        action_cat_mapping, idx_cat_mapping = create_action_cat_mapping()

        cards = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.spades, Rank.ace)
        ]

        hand = BigTwoHand(cards)

        mask = generate_action_mask(action_cat_mapping, idx_cat_mapping, hand)

        self.assertEqual(mask[0], True)
        self.assertEqual(mask[1], True)
        self.assertEqual(mask[2], True)

        self.assertEqual(mask[14], True)
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

        hand = BigTwoHand(cards)

        mask = generate_action_mask(action_cat_mapping, idx_cat_mapping, hand)

        # skip is valid
        self.assertEqual(mask[0], True)

        # single card 1 to 5 is valid
        self.assertEqual(mask[1], True)
        self.assertEqual(mask[2], True)
        self.assertEqual(mask[3], True)
        self.assertEqual(mask[4], True)
        self.assertEqual(mask[5], True)

        self.assertEqual(mask[92], True)
        self.assertEqual(mask[100], False)