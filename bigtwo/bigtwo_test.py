import unittest

from bigtwo import BigTwo
from playingcards.card import Card, Suit, Rank


class TestBigTwo(unittest.TestCase):

    def test_is_bigger_single(self):
        spade_ace = [
            Card(Suit.spades, Rank.ace)
        ]
        diamond_three = [
            Card(Suit.diamond, Rank.three)
        ]
        self.assertTrue(BigTwo.is_bigger(spade_ace, diamond_three), "Spade Ace is bigger than Diamond Three")

        spade_two = [
            Card(Suit.spades, Rank.two)
        ]

        heart_two = [
            Card(Suit.hearts, Rank.two)
        ]

        self.assertTrue(BigTwo.is_bigger(spade_two, heart_two), "Same rank but spade is higher suit")

    def test_is_bigger_double(self):
        double_queen = [
            Card(Suit.spades, Rank.queen),
            Card(Suit.hearts, Rank.queen)
        ]

        double_king = [
            Card(Suit.spades, Rank.king),
            Card(Suit.hearts, Rank.king)
        ]

        self.assertFalse(BigTwo.is_bigger(double_queen, double_king), "Double queen is not bigger than double king")

        double_higher_queen = [
            Card(Suit.diamond, Rank.queen),
            Card(Suit.spades, Rank.queen)
        ]

        double_lower_queen = [
            Card(Suit.hearts, Rank.queen),
            Card(Suit.clubs, Rank.queen)
        ]

        self.assertTrue(BigTwo.is_bigger(double_higher_queen, double_lower_queen), "First queens suit is higher")
        self.assertFalse(BigTwo.is_bigger(double_lower_queen, double_higher_queen), "First queens suit is lower")

    def test_is_bigger_straight(self):
        four_to_eight = [
            Card(Suit.hearts, Rank.seven),
            Card(Suit.clubs, Rank.four),
            Card(Suit.spades, Rank.five),
            Card(Suit.diamond, Rank.eight),
            Card(Suit.diamond, Rank.six)
        ]

        ten_to_ace = [
            Card(Suit.hearts, Rank.jack),
            Card(Suit.spades, Rank.ace),
            Card(Suit.clubs, Rank.king),
            Card(Suit.clubs, Rank.queen),
            Card(Suit.clubs, Rank.ten)
        ]

        self.assertTrue(BigTwo.is_bigger(ten_to_ace, four_to_eight), "ten_to_ace is bigger than four_to_eight straight")

        ace_to_five = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.spades, Rank.three),
            Card(Suit.diamond, Rank.four),
            Card(Suit.diamond, Rank.two),
            Card(Suit.hearts, Rank.five)
        ]

        two_to_six = [
            Card(Suit.diamond, Rank.three),
            Card(Suit.hearts, Rank.two),
            Card(Suit.spades, Rank.four),
            Card(Suit.spades, Rank.six),
            Card(Suit.spades, Rank.five),
        ]

        self.assertTrue(BigTwo.is_bigger(two_to_six, ten_to_ace), "two_to_six is bigger than ten_to_ace")
        self.assertTrue(BigTwo.is_bigger(ace_to_five, ten_to_ace), "ace_to_five is bigger than ten_to_ace")
        self.assertTrue(BigTwo.is_bigger(ace_to_five, two_to_six), "ace_to_five is bigger than two_to_six")

        lower_two_to_six = [
            Card(Suit.clubs, Rank.two),
            Card(Suit.spades, Rank.three),
            Card(Suit.clubs, Rank.four),
            Card(Suit.diamond, Rank.five),
            Card(Suit.hearts, Rank.six)
        ]

        self.assertFalse(BigTwo.is_bigger(lower_two_to_six, two_to_six), "two to six with club two is not bigger")

    def test_is_bigger_flush(self):
        heart_flush = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.hearts, Rank.six),
            Card(Suit.hearts, Rank.ten),
            Card(Suit.hearts, Rank.three),
            Card(Suit.hearts, Rank.nine)
        ]

        club_flush = [
            Card(Suit.clubs, Rank.queen),
            Card(Suit.clubs, Rank.jack),
            Card(Suit.clubs, Rank.eight),
            Card(Suit.clubs, Rank.two),
            Card(Suit.clubs, Rank.six)
        ]

        self.assertTrue(BigTwo.is_bigger(heart_flush, club_flush), "Heart flush is bigger than club flush")

        lower_heart_flush = [
            Card(Suit.hearts, Rank.king),
            Card(Suit.hearts, Rank.four),
            Card(Suit.hearts, Rank.five),
            Card(Suit.hearts, Rank.queen),
            Card(Suit.hearts, Rank.jack),
        ]

        self.assertFalse(BigTwo.is_bigger(lower_heart_flush, heart_flush), "K Flush is not bigger than A Flush")

    def test_is_bigger_full_house(self):
        ten_full_house = [
            Card(Suit.spades, Rank.ten),
            Card(Suit.hearts, Rank.ten),
            Card(Suit.clubs, Rank.ten),
            Card(Suit.spades, Rank.five),
            Card(Suit.diamond, Rank.five)
        ]

        six_full_house = [
            Card(Suit.spades, Rank.six),
            Card(Suit.hearts, Rank.six),
            Card(Suit.clubs, Rank.six),
            Card(Suit.spades, Rank.king),
            Card(Suit.diamond, Rank.king)
        ]

        self.assertTrue(BigTwo.is_bigger(ten_full_house, six_full_house),
                        "Full house ten is bigger than Full house six")

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
            Card(Suit.diamond, Rank.four)
        ]

        self.assertTrue(BigTwo.is_bigger(jack_four_of_a_kind, nine_four_of_a_kind),
                        "Jack 4 of a kind is bigger than Nine 4 of a kind")

    def test_is_bigger_straight_flush(self):
        spades_straight_flush = [
            Card(Suit.spades, Rank.ten),
            Card(Suit.spades, Rank.ace),
            Card(Suit.spades, Rank.jack),
            Card(Suit.spades, Rank.queen),
            Card(Suit.spades, Rank.king)
        ]

        heart_straight_flush = [
            Card(Suit.hearts, Rank.ten),
            Card(Suit.hearts, Rank.ace),
            Card(Suit.hearts, Rank.jack),
            Card(Suit.hearts, Rank.queen),
            Card(Suit.hearts, Rank.king)
        ]

        self.assertTrue(BigTwo.is_bigger(spades_straight_flush, heart_straight_flush),
                        "Spades straight flush is bigger than heart straight flush")

        spades_two_straight_flush = [
            Card(Suit.spades, Rank.three),
            Card(Suit.spades, Rank.two),
            Card(Suit.spades, Rank.six),
            Card(Suit.spades, Rank.five),
            Card(Suit.spades, Rank.four)
        ]

        self.assertTrue(BigTwo.is_bigger(spades_two_straight_flush, spades_straight_flush), "Spade two flush is bigger")

    def test_is_bigger_mixed_combination(self):
        straight = [
            Card(Suit.hearts, Rank.seven),
            Card(Suit.clubs, Rank.four),
            Card(Suit.spades, Rank.five),
            Card(Suit.diamond, Rank.eight),
            Card(Suit.diamond, Rank.six)
        ]

        flush = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.hearts, Rank.six),
            Card(Suit.hearts, Rank.ten),
            Card(Suit.hearts, Rank.three),
            Card(Suit.hearts, Rank.nine)
        ]

        full_house = [
            Card(Suit.spades, Rank.ten),
            Card(Suit.hearts, Rank.ten),
            Card(Suit.clubs, Rank.ten),
            Card(Suit.spades, Rank.five),
            Card(Suit.diamond, Rank.five)
        ]

        four_of_a_kind = [
            Card(Suit.spades, Rank.nine),
            Card(Suit.hearts, Rank.nine),
            Card(Suit.clubs, Rank.nine),
            Card(Suit.diamond, Rank.nine),
            Card(Suit.diamond, Rank.four)
        ]

        straight_flush = [
            Card(Suit.spades, Rank.ten),
            Card(Suit.spades, Rank.ace),
            Card(Suit.spades, Rank.jack),
            Card(Suit.spades, Rank.queen),
            Card(Suit.spades, Rank.king)
        ]

        self.assertTrue(BigTwo.is_bigger(flush, straight), "Flush bigger than straight")
        self.assertTrue(BigTwo.is_bigger(full_house, flush), "Full house bigger than flush")
        self.assertTrue(BigTwo.is_bigger(four_of_a_kind, full_house), "Four of a kind bigger than Full house")
        self.assertTrue(BigTwo.is_bigger(straight_flush, four_of_a_kind), "Straight flush bigger than 4 of a kind")

    def test_remove_card_from_hand(self):
        cards = [
            Card(Suit.diamond, Rank.king),
            Card(Suit.diamond, Rank.three)
        ]

        hand = [
            Card(Suit.hearts, Rank.king),
            Card(Suit.diamond, Rank.king),
            Card(Suit.diamond, Rank.three),
            Card(Suit.spades, Rank.three)
        ]

        result = BigTwo.remove_card_from_hand(cards, hand)

        self.assertEqual(len(result), 2)
        self.assertNotIn(Card(Suit.diamond, Rank.king), result)
        self.assertNotIn(Card(Suit.diamond, Rank.three), result)

    def test_get_five_card_type(self):
        straight = [
            Card(Suit.hearts, Rank.seven),
            Card(Suit.clubs, Rank.four),
            Card(Suit.spades, Rank.five),
            Card(Suit.diamond, Rank.eight),
            Card(Suit.diamond, Rank.six)
        ]

        flush = [
            Card(Suit.hearts, Rank.ace),
            Card(Suit.hearts, Rank.six),
            Card(Suit.hearts, Rank.ten),
            Card(Suit.hearts, Rank.three),
            Card(Suit.hearts, Rank.nine)
        ]

        full_house = [
            Card(Suit.spades, Rank.ten),
            Card(Suit.hearts, Rank.ten),
            Card(Suit.clubs, Rank.ten),
            Card(Suit.spades, Rank.five),
            Card(Suit.diamond, Rank.five)
        ]

        four_of_a_kind = [
            Card(Suit.spades, Rank.nine),
            Card(Suit.hearts, Rank.nine),
            Card(Suit.clubs, Rank.nine),
            Card(Suit.diamond, Rank.nine),
            Card(Suit.diamond, Rank.four)
        ]

        straight_flush = [
            Card(Suit.spades, Rank.ten),
            Card(Suit.spades, Rank.ace),
            Card(Suit.spades, Rank.jack),
            Card(Suit.spades, Rank.queen),
            Card(Suit.spades, Rank.king)
        ]

        self.assertEqual(BigTwo.get_five_card_type(straight), BigTwo.STRAIGHT)
        self.assertEqual(BigTwo.get_five_card_type(flush), BigTwo.FLUSH)
        self.assertEqual(BigTwo.get_five_card_type(four_of_a_kind), BigTwo.FOUR_OF_A_KIND)
        self.assertEqual(BigTwo.get_five_card_type(full_house), BigTwo.FULL_HOUSE)
        self.assertEqual(BigTwo.get_five_card_type(straight_flush), BigTwo.STRAIGHT_FLUSH)

    def test_get_five_card_type(self):
        hand = [
            Card(Suit.hearts, Rank.king),
            Card(Suit.diamond, Rank.king),
            Card(Suit.diamond, Rank.three),
            Card(Suit.spades, Rank.three),
            Card(Suit.spades, Rank.two)
        ]

        self.assertRaises(ValueError, BigTwo.get_five_card_type, hand)


if __name__ == '__main__':
    unittest.main()
