import unittest

from bigtwo.bigtwo import (
    BigTwo,
    BigTwoHand,
    find_combinations_from_cards,
)
from gamerunner.random_bot import RandomBot
from playingcards.card import Card, Suit, Rank


class TestBigTwo(unittest.TestCase):
    def test_is_bigger_single(self):
        spade_ace = [Card(Suit.spades, Rank.ace)]
        diamond_three = [Card(Suit.diamond, Rank.three)]
        self.assertTrue(
            BigTwo.is_bigger(spade_ace, diamond_three),
            "Spade Ace is bigger than Diamond Three",
        )

        spade_two = [Card(Suit.spades, Rank.two)]

        heart_two = [Card(Suit.hearts, Rank.two)]

        self.assertTrue(
            BigTwo.is_bigger(spade_two, heart_two), "Same rank but spade is higher suit"
        )

        spade_ten = [Card(Suit.spades, Rank.ten)]

        spade_queen = [Card(Suit.spades, Rank.queen)]

        self.assertTrue(
            BigTwo.is_bigger(spade_queen, spade_ten),
            "Same rank but spade is higher suit",
        )

    def test_is_bigger_double(self):
        double_queen = [Card(Suit.spades, Rank.queen), Card(Suit.hearts, Rank.queen)]

        double_king = [Card(Suit.spades, Rank.king), Card(Suit.hearts, Rank.king)]

        self.assertFalse(
            BigTwo.is_bigger(double_queen, double_king),
            "Double queen is not bigger than double king",
        )

        double_higher_queen = [
            Card(Suit.diamond, Rank.queen),
            Card(Suit.spades, Rank.queen),
        ]

        double_lower_queen = [
            Card(Suit.hearts, Rank.queen),
            Card(Suit.clubs, Rank.queen),
        ]

        self.assertTrue(
            BigTwo.is_bigger(double_higher_queen, double_lower_queen),
            "First queens suit is higher",
        )
        self.assertFalse(
            BigTwo.is_bigger(double_lower_queen, double_higher_queen),
            "First queens suit is lower",
        )

    def test_is_bigger_straight(self):
        four_to_eight = [
            Card(Suit.hearts, Rank.seven),
            Card(Suit.clubs, Rank.four),
            Card(Suit.spades, Rank.five),
            Card(Suit.diamond, Rank.eight),
            Card(Suit.diamond, Rank.six),
        ]

        ten_to_ace = [
            Card(Suit.hearts, Rank.jack),
            Card(Suit.spades, Rank.ace),
            Card(Suit.clubs, Rank.king),
            Card(Suit.clubs, Rank.queen),
            Card(Suit.clubs, Rank.ten),
        ]

        self.assertTrue(
            BigTwo.is_bigger(ten_to_ace, four_to_eight),
            "ten_to_ace is bigger than four_to_eight straight",
        )

        ace_to_five = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.spades, Rank.three),
            Card(Suit.diamond, Rank.four),
            Card(Suit.diamond, Rank.two),
            Card(Suit.hearts, Rank.five),
        ]

        two_to_six = [
            Card(Suit.diamond, Rank.three),
            Card(Suit.hearts, Rank.two),
            Card(Suit.spades, Rank.four),
            Card(Suit.spades, Rank.six),
            Card(Suit.spades, Rank.five),
        ]

        self.assertTrue(
            BigTwo.is_bigger(two_to_six, ten_to_ace),
            "two_to_six is bigger than ten_to_ace",
        )
        self.assertTrue(
            BigTwo.is_bigger(ace_to_five, ten_to_ace),
            "ace_to_five is bigger than ten_to_ace",
        )
        self.assertTrue(
            BigTwo.is_bigger(ace_to_five, two_to_six),
            "ace_to_five is bigger than two_to_six",
        )

        lower_two_to_six = [
            Card(Suit.clubs, Rank.two),
            Card(Suit.spades, Rank.three),
            Card(Suit.clubs, Rank.four),
            Card(Suit.diamond, Rank.five),
            Card(Suit.hearts, Rank.six),
        ]

        self.assertFalse(
            BigTwo.is_bigger(lower_two_to_six, two_to_six),
            "two to six with club two is not bigger",
        )

    def test_is_bigger_flush(self):
        heart_flush = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.hearts, Rank.six),
            Card(Suit.hearts, Rank.ten),
            Card(Suit.hearts, Rank.three),
            Card(Suit.hearts, Rank.nine),
        ]

        club_flush = [
            Card(Suit.clubs, Rank.queen),
            Card(Suit.clubs, Rank.jack),
            Card(Suit.clubs, Rank.eight),
            Card(Suit.clubs, Rank.two),
            Card(Suit.clubs, Rank.six),
        ]

        self.assertTrue(
            BigTwo.is_bigger(heart_flush, club_flush),
            "Heart flush is bigger than club flush",
        )

        lower_heart_flush = [
            Card(Suit.hearts, Rank.king),
            Card(Suit.hearts, Rank.four),
            Card(Suit.hearts, Rank.five),
            Card(Suit.hearts, Rank.queen),
            Card(Suit.hearts, Rank.jack),
        ]

        self.assertFalse(
            BigTwo.is_bigger(lower_heart_flush, heart_flush),
            "K Flush is not bigger than A Flush",
        )

    def test_is_bigger_full_house(self):
        ten_full_house = [
            Card(Suit.spades, Rank.ten),
            Card(Suit.hearts, Rank.ten),
            Card(Suit.clubs, Rank.ten),
            Card(Suit.spades, Rank.five),
            Card(Suit.diamond, Rank.five),
        ]

        six_full_house = [
            Card(Suit.spades, Rank.six),
            Card(Suit.hearts, Rank.six),
            Card(Suit.clubs, Rank.six),
            Card(Suit.spades, Rank.king),
            Card(Suit.diamond, Rank.king),
        ]

        self.assertTrue(
            BigTwo.is_bigger(ten_full_house, six_full_house),
            "Full house ten is bigger than Full house six",
        )

    def test_is_bigger_four_of_a_kind(self):
        jack_four_of_a_kind = [
            Card(Suit.spades, Rank.jack),
            Card(Suit.hearts, Rank.jack),
            Card(Suit.clubs, Rank.jack),
            Card(Suit.diamond, Rank.jack),
            Card(Suit.diamond, Rank.three),
        ]

        nine_four_of_a_kind = [
            Card(Suit.spades, Rank.nine),
            Card(Suit.hearts, Rank.nine),
            Card(Suit.clubs, Rank.nine),
            Card(Suit.diamond, Rank.nine),
            Card(Suit.diamond, Rank.four),
        ]

        self.assertTrue(
            BigTwo.is_bigger(jack_four_of_a_kind, nine_four_of_a_kind),
            "Jack 4 of a kind is bigger than Nine 4 of a kind",
        )

    def test_is_bigger_straight_flush(self):
        spades_straight_flush = [
            Card(Suit.spades, Rank.ten),
            Card(Suit.spades, Rank.ace),
            Card(Suit.spades, Rank.jack),
            Card(Suit.spades, Rank.queen),
            Card(Suit.spades, Rank.king),
        ]

        heart_straight_flush = [
            Card(Suit.hearts, Rank.ten),
            Card(Suit.hearts, Rank.ace),
            Card(Suit.hearts, Rank.jack),
            Card(Suit.hearts, Rank.queen),
            Card(Suit.hearts, Rank.king),
        ]

        self.assertTrue(
            BigTwo.is_bigger(spades_straight_flush, heart_straight_flush),
            "Spades straight flush is bigger than heart straight flush",
        )

        spades_two_straight_flush = [
            Card(Suit.spades, Rank.three),
            Card(Suit.spades, Rank.two),
            Card(Suit.spades, Rank.six),
            Card(Suit.spades, Rank.five),
            Card(Suit.spades, Rank.four),
        ]

        self.assertTrue(
            BigTwo.is_bigger(spades_two_straight_flush, spades_straight_flush),
            "Spade two flush is bigger",
        )

    def test_is_bigger_mixed_combination(self):
        straight = [
            Card(Suit.hearts, Rank.seven),
            Card(Suit.clubs, Rank.four),
            Card(Suit.spades, Rank.five),
            Card(Suit.diamond, Rank.eight),
            Card(Suit.diamond, Rank.six),
        ]

        flush = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.hearts, Rank.six),
            Card(Suit.hearts, Rank.ten),
            Card(Suit.hearts, Rank.three),
            Card(Suit.hearts, Rank.nine),
        ]

        full_house = [
            Card(Suit.spades, Rank.ten),
            Card(Suit.hearts, Rank.ten),
            Card(Suit.clubs, Rank.ten),
            Card(Suit.spades, Rank.five),
            Card(Suit.diamond, Rank.five),
        ]

        four_of_a_kind = [
            Card(Suit.spades, Rank.nine),
            Card(Suit.hearts, Rank.nine),
            Card(Suit.clubs, Rank.nine),
            Card(Suit.diamond, Rank.nine),
            Card(Suit.diamond, Rank.four),
        ]

        straight_flush = [
            Card(Suit.spades, Rank.ten),
            Card(Suit.spades, Rank.ace),
            Card(Suit.spades, Rank.jack),
            Card(Suit.spades, Rank.queen),
            Card(Suit.spades, Rank.king),
        ]

        self.assertTrue(BigTwo.is_bigger(flush, straight), "Flush bigger than straight")
        self.assertTrue(
            BigTwo.is_bigger(full_house, flush), "Full house bigger than flush"
        )
        self.assertTrue(
            BigTwo.is_bigger(four_of_a_kind, full_house),
            "Four of a kind bigger than Full house",
        )
        self.assertTrue(
            BigTwo.is_bigger(straight_flush, four_of_a_kind),
            "Straight flush bigger than 4 of a kind",
        )

    def test_get_five_card_type(self):
        straight = [
            Card(Suit.hearts, Rank.seven),
            Card(Suit.clubs, Rank.four),
            Card(Suit.spades, Rank.five),
            Card(Suit.diamond, Rank.eight),
            Card(Suit.diamond, Rank.six),
        ]

        flush = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.hearts, Rank.six),
            Card(Suit.hearts, Rank.ten),
            Card(Suit.hearts, Rank.three),
            Card(Suit.hearts, Rank.nine),
        ]

        full_house = [
            Card(Suit.spades, Rank.ten),
            Card(Suit.hearts, Rank.ten),
            Card(Suit.clubs, Rank.ten),
            Card(Suit.spades, Rank.five),
            Card(Suit.diamond, Rank.five),
        ]

        four_of_a_kind = [
            Card(Suit.spades, Rank.nine),
            Card(Suit.hearts, Rank.nine),
            Card(Suit.clubs, Rank.nine),
            Card(Suit.diamond, Rank.nine),
            Card(Suit.diamond, Rank.four),
        ]

        straight_flush = [
            Card(Suit.spades, Rank.ten),
            Card(Suit.spades, Rank.ace),
            Card(Suit.spades, Rank.jack),
            Card(Suit.spades, Rank.queen),
            Card(Suit.spades, Rank.king),
        ]

        self.assertEqual(BigTwo.get_five_card_type(straight), BigTwo.STRAIGHT)
        self.assertEqual(BigTwo.get_five_card_type(flush), BigTwo.FLUSH)
        self.assertEqual(
            BigTwo.get_five_card_type(four_of_a_kind), BigTwo.FOUR_OF_A_KIND
        )
        self.assertEqual(BigTwo.get_five_card_type(full_house), BigTwo.FULL_HOUSE)
        self.assertEqual(
            BigTwo.get_five_card_type(straight_flush), BigTwo.STRAIGHT_FLUSH
        )

    def test_get_five_card_type_err(self):
        hand = [
            Card(Suit.hearts, Rank.king),
            Card(Suit.diamond, Rank.king),
            Card(Suit.diamond, Rank.three),
            Card(Suit.spades, Rank.three),
            Card(Suit.spades, Rank.two),
        ]

        self.assertRaises(ValueError, BigTwo.get_five_card_type, hand)

    def test_is_valid_card_combination_single(self):
        ace = Card(Suit.spades, Rank.ace)

        is_valid = BigTwo.is_valid_card_combination([ace])

        self.assertTrue(is_valid)

    def test_is_valid_combination_double(self):
        valid_pair = [Card(Suit.spades, Rank.ace), Card(Suit.hearts, Rank.ace)]

        is_valid = BigTwo.is_valid_card_combination(valid_pair)

        self.assertTrue(is_valid)

    def test_is_valid_combination_double_err(self):
        invalid_pair = [Card(Suit.spades, Rank.ace), Card(Suit.hearts, Rank.two)]

        is_valid = BigTwo.is_valid_card_combination(invalid_pair)

        self.assertFalse(is_valid)

    def test_is_valid_combination_triple(self):
        invalid_pair = [
            Card(Suit.spades, Rank.ace),
            Card(Suit.hearts, Rank.two),
            Card(Suit.hearts, Rank.three),
        ]

        is_valid = BigTwo.is_valid_card_combination(invalid_pair)

        self.assertFalse(is_valid)

    def test_is_valid_combination_four(self):
        invalid_pair = [
            Card(Suit.spades, Rank.ace),
            Card(Suit.hearts, Rank.two),
            Card(Suit.hearts, Rank.three),
            Card(Suit.hearts, Rank.four),
        ]

        is_valid = BigTwo.is_valid_card_combination(invalid_pair)

        self.assertFalse(is_valid)

    def test_is_valid_combination_invalid_straight(self):
        invalid_pair = [
            Card(Suit.spades, Rank.ace),
            Card(Suit.hearts, Rank.two),
            Card(Suit.hearts, Rank.three),
            Card(Suit.hearts, Rank.four),
            Card(Suit.hearts, Rank.six),
        ]

        is_valid = BigTwo.is_valid_card_combination(invalid_pair)

        self.assertFalse(is_valid)

    def test_is_valid_combination_valid_straight(self):
        valid_pair = [
            Card(Suit.spades, Rank.ace),
            Card(Suit.hearts, Rank.two),
            Card(Suit.hearts, Rank.three),
            Card(Suit.hearts, Rank.four),
            Card(Suit.hearts, Rank.five),
        ]

        is_valid = BigTwo.is_valid_card_combination(valid_pair)

        self.assertTrue(is_valid)

    def test_is_valid_combination_valid_full_house(self):
        invalid_pair = [
            Card(Suit.spades, Rank.ace),
            Card(Suit.hearts, Rank.ace),
            Card(Suit.clubs, Rank.ace),
            Card(Suit.hearts, Rank.four),
            Card(Suit.spades, Rank.four),
        ]

        is_valid = BigTwo.is_valid_card_combination(invalid_pair)

        self.assertTrue(is_valid)

    def test_bigtwo_hand_remove_cards_comb_left(self):
        hand = [
            Card(Suit.spades, Rank.ace),
            Card(Suit.hearts, Rank.ace),
            Card(Suit.clubs, Rank.ace),
            Card(Suit.hearts, Rank.four),
            Card(Suit.spades, Rank.four),
            Card(Suit.clubs, Rank.four),
            Card(Suit.clubs, Rank.ten),
        ]

        bigtwo_hand = BigTwoHand(hand)

        expected_comb = (
            Card(Suit.spades, Rank.ace),
            Card(Suit.hearts, Rank.ace),
            Card(Suit.clubs, Rank.ace),
            Card(Suit.hearts, Rank.four),
            Card(Suit.spades, Rank.four),
        )

        self.assertEqual(len(bigtwo_hand.combinations), 1)
        self.assertIn(BigTwo.FULL_HOUSE, bigtwo_hand.combinations)
        self.assertEqual(len(bigtwo_hand.combinations[BigTwo.FULL_HOUSE]), 6)
        self.assertIn(expected_comb, bigtwo_hand.combinations[BigTwo.FULL_HOUSE])

        bigtwo_hand.remove_cards([Card(Suit.spades, Rank.ace)])

        self.assertEqual(len(bigtwo_hand), 6)
        self.assertEqual(len(bigtwo_hand.combinations), 1)
        self.assertIn(BigTwo.FULL_HOUSE, bigtwo_hand.combinations)
        self.assertEqual(len(bigtwo_hand.combinations[BigTwo.FULL_HOUSE]), 1)

    def test_bigtwo_hand_remove_cards_no_comb_left(self):
        hand = [
            Card(Suit.spades, Rank.ace),
            Card(Suit.hearts, Rank.ace),
            Card(Suit.clubs, Rank.ace),
            Card(Suit.hearts, Rank.four),
            Card(Suit.spades, Rank.four),
            Card(Suit.clubs, Rank.ten),
        ]

        bigtwo_hand = BigTwoHand(hand)

        expected_comb = (
            Card(Suit.spades, Rank.ace),
            Card(Suit.hearts, Rank.ace),
            Card(Suit.clubs, Rank.ace),
            Card(Suit.hearts, Rank.four),
            Card(Suit.spades, Rank.four),
        )

        self.assertEqual(len(bigtwo_hand.combinations), 1)
        self.assertIn(BigTwo.FULL_HOUSE, bigtwo_hand.combinations)
        self.assertIn(expected_comb, bigtwo_hand.combinations[BigTwo.FULL_HOUSE])

        bigtwo_hand.remove_cards([Card(Suit.spades, Rank.ace)])

        self.assertEqual(len(bigtwo_hand), 5)
        self.assertEqual(len(bigtwo_hand.combinations), 0)

    def test_play_diamond_three(self):
        env = BigTwo()
        obs = env.reset()

        self.assertEqual(len(obs.your_hands), 13)
        self.assertEqual(obs.num_card_per_player, [13, 13, 13])
        self.assertEqual(obs.last_player_played, 5)

        my_player_number = obs.current_player

        raw_action = []
        for idx, card in enumerate(obs.your_hands):
            if card.rank == Rank.three and card.suit == Suit.diamond:
                raw_action.append(1)
                continue

            raw_action.append(0)

        new_obs, reward, done = env.step(raw_action)

        self.assertFalse(done)

        self.assertEqual(len(new_obs.your_hands), 12)
        self.assertEqual(new_obs.last_cards_played, [Card(Suit.diamond, Rank.three)])
        self.assertEqual(new_obs.last_player_played, my_player_number)

        curr_obs = env.get_current_player_obs()
        self.assertEqual(curr_obs.num_card_per_player, [13, 13, 12])

    def test_playable_cards_empty_state(self):
        env = BigTwo()

        hand = [Card(Suit.hearts, Rank.ace), Card(Suit.diamond, Rank.three)]

        playable = env.have_playable_cards([], BigTwoHand(hand))

        self.assertTrue(playable)

    def test_playable_cards_single_true(self):
        env = BigTwo()

        hand = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.clubs, Rank.four),
        ]

        playable = env.have_playable_cards(
            [Card(Suit.spades, Rank.ten)], BigTwoHand(hand)
        )

        self.assertTrue(playable, "ace is higher than ten")

        hand = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.clubs, Rank.four),
        ]

        playable = env.have_playable_cards(
            [Card(Suit.diamond, Rank.four)], BigTwoHand(hand)
        )

        self.assertTrue(playable, "club 4 is higher than diamond 4")

    def test_playable_cards_single_false(self):
        env = BigTwo()

        hand = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.clubs, Rank.four),
        ]

        playable = env.have_playable_cards(
            [Card(Suit.spades, Rank.two)], BigTwoHand(hand)
        )

        self.assertFalse(playable, "two is higher than all cards")

        hand = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.clubs, Rank.four),
        ]

        playable = env.have_playable_cards(
            [Card(Suit.spades, Rank.ace)], BigTwoHand(hand)
        )

        self.assertFalse(playable, "spades ace is higher than hearts ace")

    def test_playable_cards_double_true(self):
        env = BigTwo()

        hand = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.spades, Rank.ace),
        ]

        target = [
            Card(Suit.diamond, Rank.ace),
            Card(Suit.clubs, Rank.ace),
        ]

        playable = env.have_playable_cards(target, BigTwoHand(hand))

        self.assertTrue(playable)

    def test_playable_cards_double_false(self):
        env = BigTwo()

        hand = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.diamond, Rank.ace),
        ]

        target = [
            Card(Suit.spades, Rank.ace),
            Card(Suit.clubs, Rank.ace),
        ]

        playable = env.have_playable_cards(target, BigTwoHand(hand))

        self.assertFalse(playable)

    def test_playable_cards_combination_true(self):
        env = BigTwo()

        hand = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.diamond, Rank.ace),
            Card(Suit.spades, Rank.ace),
            Card(Suit.spades, Rank.ten),
            Card(Suit.clubs, Rank.ten),
            Card(Suit.clubs, Rank.seven),
        ]

        target = [
            Card(Suit.spades, Rank.queen),
            Card(Suit.spades, Rank.jack),
            Card(Suit.spades, Rank.five),
            Card(Suit.spades, Rank.four),
            Card(Suit.spades, Rank.seven),
        ]

        playable = env.have_playable_cards(target, BigTwoHand(hand))

        self.assertTrue(playable)

        hand = [
            Card(Suit.spades, Rank.two),
            Card(Suit.spades, Rank.ten),
            Card(Suit.spades, Rank.nine),
            Card(Suit.spades, Rank.eight),
            Card(Suit.spades, Rank.three),
        ]

        target = [
            Card(Suit.spades, Rank.queen),
            Card(Suit.spades, Rank.jack),
            Card(Suit.spades, Rank.five),
            Card(Suit.spades, Rank.four),
            Card(Suit.spades, Rank.seven),
        ]

        playable = env.have_playable_cards(target, BigTwoHand(hand))

        self.assertTrue(playable)

    def test_playable_cards_combination_false(self):
        env = BigTwo()

        hand = [
            Card(Suit.spades, Rank.queen),
            Card(Suit.spades, Rank.jack),
            Card(Suit.spades, Rank.five),
            Card(Suit.spades, Rank.four),
            Card(Suit.spades, Rank.seven),
            Card(Suit.clubs, Rank.seven),
        ]

        target = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.diamond, Rank.ace),
            Card(Suit.spades, Rank.ace),
            Card(Suit.spades, Rank.ten),
            Card(Suit.clubs, Rank.ten),
        ]

        playable = env.have_playable_cards(target, BigTwoHand(hand))

        self.assertFalse(playable)

        hand = [
            Card(Suit.clubs, Rank.five),
            Card(Suit.spades, Rank.six),
            Card(Suit.hearts, Rank.nine),
            Card(Suit.spades, Rank.eight),
            Card(Suit.diamond, Rank.seven),
        ]

        target = [
            Card(Suit.spades, Rank.queen),
            Card(Suit.spades, Rank.jack),
            Card(Suit.spades, Rank.ace),
            Card(Suit.hearts, Rank.king),
            Card(Suit.spades, Rank.ten),
        ]

        playable = env.have_playable_cards(target, BigTwoHand(hand))

        self.assertFalse(playable)

    def test_calculate_valid_pairs(self):
        cards = [
            Card(Suit.spades, Rank.two),
            Card(Suit.spades, Rank.ten),
            Card(Suit.hearts, Rank.ten),
            Card(Suit.clubs, Rank.two),
        ]

        hand = BigTwoHand(cards)

        self.assertEqual(len(hand.pairs), 2)
        self.assertIn(
            (Card(Suit.spades, Rank.two), Card(Suit.clubs, Rank.two)), hand.pairs
        )
        self.assertIn(
            (Card(Suit.spades, Rank.ten), Card(Suit.hearts, Rank.ten)), hand.pairs
        )

    def test_calculate_remove_valid_pairs(self):
        cards = [
            Card(Suit.spades, Rank.two),
            Card(Suit.spades, Rank.ten),
            Card(Suit.hearts, Rank.ten),
            Card(Suit.clubs, Rank.two),
        ]

        hand = BigTwoHand(cards)

        self.assertEqual(len(hand.pairs), 2)
        self.assertIn(
            (Card(Suit.spades, Rank.two), Card(Suit.clubs, Rank.two)), hand.pairs
        )
        self.assertIn(
            (Card(Suit.spades, Rank.ten), Card(Suit.hearts, Rank.ten)), hand.pairs
        )

        hand.remove_cards([Card(Suit.spades, Rank.two)])

        self.assertEqual(len(hand.pairs), 1)

        self.assertIn(
            (Card(Suit.spades, Rank.ten), Card(Suit.hearts, Rank.ten)), hand.pairs
        )

        if __name__ == "__main__":
            unittest.main()

    def test_find_combinations_straight(self):
        cards = [
            Card(Suit.spades, Rank.two),
            Card(Suit.spades, Rank.three),
            Card(Suit.hearts, Rank.four),
            Card(Suit.clubs, Rank.five),
            Card(Suit.clubs, Rank.six),
        ]
        combinations = find_combinations_from_cards(cards)

        self.assertIn(BigTwo.STRAIGHT, combinations)
        self.assertEqual(len(combinations[BigTwo.STRAIGHT]), 1)
        self.assertTupleEqual(combinations[BigTwo.STRAIGHT][0], tuple(cards))

        cards = [
            Card(Suit.spades, Rank.two),
            Card(Suit.spades, Rank.three),
            Card(Suit.hearts, Rank.four),
            Card(Suit.clubs, Rank.five),
            Card(Suit.clubs, Rank.six),
            Card(Suit.clubs, Rank.seven),
        ]
        combinations = find_combinations_from_cards(cards)

        self.assertIn(BigTwo.STRAIGHT, combinations)
        self.assertEqual(len(combinations[BigTwo.STRAIGHT]), 2)
        self.assertTupleEqual(combinations[BigTwo.STRAIGHT][0], tuple(cards[:5]))
        self.assertTupleEqual(combinations[BigTwo.STRAIGHT][1], tuple(cards[1:]))

    def test_find_combinations_straight_special(self):
        cards = [
            Card(Suit.spades, Rank.ten),
            Card(Suit.spades, Rank.jack),
            Card(Suit.hearts, Rank.queen),
            Card(Suit.clubs, Rank.king),
            Card(Suit.clubs, Rank.ace),
            Card(Suit.clubs, Rank.seven),
        ]
        combinations = find_combinations_from_cards(cards)

        self.assertIn(BigTwo.STRAIGHT, combinations)
        self.assertEqual(len(combinations[BigTwo.STRAIGHT]), 1)
        self.assertTupleEqual(combinations[BigTwo.STRAIGHT][0], tuple(cards[:5]))

    def test_find_combinations_straight_not_found(self):
        cards = [
            Card(Suit.spades, Rank.ten),
            Card(Suit.spades, Rank.jack),
            Card(Suit.hearts, Rank.six),
            Card(Suit.clubs, Rank.king),
            Card(Suit.clubs, Rank.ace),
            Card(Suit.clubs, Rank.seven),
        ]
        combinations = find_combinations_from_cards(cards)

        self.assertNotIn(BigTwo.STRAIGHT, combinations)
        self.assertEqual(len(combinations), 0)

    def test_find_combinations_flush(self):
        cards = [
            Card(Suit.spades, Rank.ten),
            Card(Suit.spades, Rank.jack),
            Card(Suit.spades, Rank.six),
            Card(Suit.spades, Rank.king),
            Card(Suit.spades, Rank.ace),
            Card(Suit.clubs, Rank.seven),
            Card(Suit.spades, Rank.two),
        ]
        combinations = find_combinations_from_cards(cards)

        self.assertIn(BigTwo.FLUSH, combinations)
        self.assertEqual(len(combinations[BigTwo.FLUSH]), 6)

    def test_find_combinations_straight_flush(self):
        cards = [
            Card(Suit.clubs, Rank.seven),
            Card(Suit.spades, Rank.ten),
            Card(Suit.spades, Rank.jack),
            Card(Suit.spades, Rank.queen),
            Card(Suit.spades, Rank.king),
            Card(Suit.spades, Rank.ace),
        ]
        combinations = find_combinations_from_cards(cards)

        self.assertIn(BigTwo.STRAIGHT_FLUSH, combinations)
        self.assertNotIn(BigTwo.STRAIGHT, combinations)
        self.assertEqual(len(combinations[BigTwo.STRAIGHT_FLUSH]), 1)
        self.assertTupleEqual(combinations[BigTwo.STRAIGHT_FLUSH][0], tuple(cards[1:]))

    def test_find_combinations_full_house(self):
        cards = [
            Card(Suit.clubs, Rank.ten),
            Card(Suit.spades, Rank.ten),
            Card(Suit.hearts, Rank.ten),
            Card(Suit.spades, Rank.queen),
            Card(Suit.hearts, Rank.queen),
        ]
        combinations = find_combinations_from_cards(cards)

        self.assertIn(BigTwo.FULL_HOUSE, combinations)
        self.assertEqual(len(combinations[BigTwo.FULL_HOUSE]), 1)
        self.assertTupleEqual(combinations[BigTwo.FULL_HOUSE][0], tuple(cards))

    def test_find_combinations_multiple_full_house(self):
        cards = [
            Card(Suit.clubs, Rank.ten),
            Card(Suit.spades, Rank.ten),
            Card(Suit.hearts, Rank.ten),
            Card(Suit.spades, Rank.queen),
            Card(Suit.hearts, Rank.queen),
            Card(Suit.diamond, Rank.queen),
        ]
        combinations = find_combinations_from_cards(cards)

        self.assertIn(BigTwo.FULL_HOUSE, combinations)
        self.assertEqual(len(combinations[BigTwo.FULL_HOUSE]), 6)

    def test_find_combinations_four_of_a_kind(self):
        cards = [
            Card(Suit.clubs, Rank.ten),
            Card(Suit.spades, Rank.ten),
            Card(Suit.hearts, Rank.ten),
            Card(Suit.diamond, Rank.ten),
            Card(Suit.spades, Rank.queen),
            Card(Suit.hearts, Rank.queen),
        ]
        combinations = find_combinations_from_cards(cards)

        self.assertIn(BigTwo.FOUR_OF_A_KIND, combinations)
        self.assertEqual(len(combinations[BigTwo.FOUR_OF_A_KIND]), 2)
        self.assertTupleEqual(
            combinations[BigTwo.FOUR_OF_A_KIND][0],
            tuple(
                [
                    Card(Suit.clubs, Rank.ten),
                    Card(Suit.spades, Rank.ten),
                    Card(Suit.hearts, Rank.ten),
                    Card(Suit.diamond, Rank.ten),
                    Card(Suit.spades, Rank.queen),
                ]
            ),
        )

        self.assertTupleEqual(
            combinations[BigTwo.FOUR_OF_A_KIND][1],
            tuple(
                [
                    Card(Suit.clubs, Rank.ten),
                    Card(Suit.spades, Rank.ten),
                    Card(Suit.hearts, Rank.ten),
                    Card(Suit.diamond, Rank.ten),
                    Card(Suit.hearts, Rank.queen),
                ]
            ),
        )

        self.assertIn(BigTwo.FULL_HOUSE, combinations)
        self.assertEqual(len(combinations[BigTwo.FULL_HOUSE]), 4)

    def test_is_straight_false(self):
        cards = [
            Card(Suit.clubs, Rank.jack),
            Card(Suit.spades, Rank.queen),
            Card(Suit.hearts, Rank.king),
            Card(Suit.spades, Rank.ace),
            Card(Suit.hearts, Rank.two),
        ]
        self.assertFalse(BigTwo.is_straight(cards))

        cards = [
            Card(Suit.clubs, Rank.ten),
            Card(Suit.spades, Rank.ten),
            Card(Suit.hearts, Rank.ten),
            Card(Suit.spades, Rank.queen),
            Card(Suit.hearts, Rank.queen),
        ]
        self.assertFalse(BigTwo.is_straight(cards))

    def test_is_straight_true(self):
        cards = [
            Card(Suit.clubs, Rank.jack),
            Card(Suit.spades, Rank.queen),
            Card(Suit.hearts, Rank.king),
            Card(Suit.spades, Rank.ace),
            Card(Suit.hearts, Rank.ten),
        ]
        self.assertTrue(BigTwo.is_straight(cards))

        cards = [
            Card(Suit.clubs, Rank.ace),
            Card(Suit.spades, Rank.two),
            Card(Suit.hearts, Rank.three),
            Card(Suit.spades, Rank.four),
            Card(Suit.hearts, Rank.five),
        ]
        self.assertTrue(BigTwo.is_straight(cards))

        cards = [
            Card(Suit.clubs, Rank.six),
            Card(Suit.spades, Rank.two),
            Card(Suit.hearts, Rank.three),
            Card(Suit.spades, Rank.four),
            Card(Suit.hearts, Rank.five),
        ]
        self.assertTrue(BigTwo.is_straight(cards))

    def test_step_when_game_done(self):
        env = BigTwo()

        while True:
            obs = env.get_current_player_obs()
            bot = RandomBot()
            raw_action = bot.action(obs)
            _, _, done = env.step(raw_action)

            if done:
                break

        obs = env.get_current_player_obs()

        current_player_number = obs.current_player

        # play the first card
        raw_action = [0] * 13
        raw_action[0] = 1

        _, _, done = env.step(raw_action)
        self.assertTrue(done)

        # assert the current player number didn't change
        self.assertEqual(current_player_number, env.current_player)
