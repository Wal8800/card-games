import unittest
from playingcards import Card, Suit, Rank, Deck
from bigtwo import BigTwo
from gamerunner import Bot
import random


class TestBigTwo(unittest.TestCase):

    @staticmethod
    def create_big_two_game() -> BigTwo:
        deck = Deck()
        player_list = []
        for i in range(BigTwo.number_of_players()):
            player_list.append(Bot(str(i)))

        return BigTwo(player_list, deck)

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

        self.assertTrue(BigTwo.is_bigger(ten_full_house, six_full_house), "Full house ten is bigger than Full house six")

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

        self.assertTrue(BigTwo.is_bigger(jack_four_of_a_kind, nine_four_of_a_kind), "Jack 4 of a kind is bigger than Nine 4 of a kind")

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

        self.assertTrue(BigTwo.is_bigger(spades_straight_flush, heart_straight_flush), "Spades straight flush is bigger than heart straight flush")

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

    def test_step_first_action(self):
        big_two_game = TestBigTwo.create_big_two_game()
        current_player, current_player_number = big_two_game.get_current_player()
        obs = big_two_game.current_observation(current_player_number)
        random_card = random.choice(obs.your_hands)
        result_obs, reward, done = big_two_game.step([random_card])

        self.assertFalse(done)
        self.assertEqual(reward, -1)

        self.assertNotIn(random_card, result_obs.your_hands)
        self.assertEqual(result_obs.num_card_per_player, [13, 13, 13])
        self.assertEqual(result_obs.cards_played, [[random_card]])

        next_player = current_player_number + 1
        if next_player > 3:
            next_player = 0

        self.assertEqual(big_two_game.current_player, next_player)

    def test_step_skip_action(self):
        big_two_game = TestBigTwo.create_big_two_game()

        # first player plays a card first
        current_player, current_player_number = big_two_game.get_current_player()
        obs = big_two_game.current_observation(current_player_number)
        random_card = random.choice(obs.your_hands)
        _, reward, done = big_two_game.step([random_card])

        self.assertFalse(done)
        self.assertEqual(reward, -1)

        # second player skip a turn
        second_player, second_player_number = big_two_game.get_current_player()
        second_player_obs = big_two_game.current_observation(second_player_number)
        next_player = current_player_number + 1
        if next_player > 3:
            next_player = 0

        self.assertEqual(second_player_number, next_player)

        result_obs, reward, done = big_two_game.step([])
        self.assertFalse(done)
        self.assertEqual(reward, 0)

        # asserting nothing changed since the player skipp the turn
        self.assertEqual(second_player_obs.your_hands, result_obs.your_hands)
        self.assertEqual(second_player_obs.cards_played, result_obs.cards_played)
        self.assertEqual(second_player_obs.num_card_per_player, result_obs.num_card_per_player)

        # asserting that it moved to another player turns
        next_next_player = second_player_number + 1
        if next_next_player > 3:
            next_next_player = 0
        self.assertEqual(big_two_game.current_player, next_next_player)


if __name__ == '__main__':
    unittest.main()
