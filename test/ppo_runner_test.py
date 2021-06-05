import unittest

from bigtwo.bigtwo import BigTwo, BigTwoHand, BigTwoObservation
from gamerunner.ppo_bot import RandomPPOBot, PPOAction
from gamerunner.ppo_runner import SinglePlayerWrapper
from playingcards.card import Suit, Rank, Card


class PlayFirstCardBot:
    def action(self, observation: BigTwoObservation) -> PPOAction:
        raw = [1 if i == 0 else 0 for i in range(13)]

        result = PPOAction(
            transformed_obs=None,
            raw_action=raw,
            action_cat=None,
            action_mask=None,
            logp=None,
            cards=None,
        )
        return result


class TestSinglePlayerWrapper(unittest.TestCase):
    def test_reset_start(self):
        env = BigTwo()

        opponents_bots = [RandomPPOBot() for _ in range(3)]
        wrapped_env = SinglePlayerWrapper(env, 0, opponents_bots)
        obs = wrapped_env.reset_and_start()

        self.assertEqual(obs.current_player, 0)
        self.assertLessEqual(env.get_hands_played(), 3)
        self.assertEqual(len(obs.your_hands), 13)

    def test_step_won_game(self):
        # how to do stateful object unit test the best practice??
        env = BigTwo()
        env.player_hands = [
            BigTwoHand([Card(Suit.spades, Rank.ace)]),
            BigTwoHand([]),
            BigTwoHand([]),
            BigTwoHand([]),
        ]
        env.state = [(3, [Card(Suit.spades, Rank.six)])]
        env.player_last_played = 3
        env.current_player = 0

        opponents_bots = [RandomPPOBot() for _ in range(3)]
        wrapped_env = SinglePlayerWrapper(env, 0, opponents_bots)

        # select the first card, spade ace
        raw_action = [1 if i == 0 else 0 for i in range(13)]

        action = PPOAction(
            transformed_obs=None,
            raw_action=raw_action,
            action_cat=None,
            action_mask=None,
            logp=None,
            cards=None,
        )

        _, reward, done = wrapped_env.step(action)
        self.assertTrue(done)
        self.assertEqual(reward, 1000)

    def test_step_lost_game(self):
        # how to do stateful object unit test the best practice??
        env = BigTwo()
        env.player_hands = [
            BigTwoHand([Card(Suit.spades, Rank.seven), Card(Suit.spades, Rank.ace)]),
            BigTwoHand([Card(Suit.spades, Rank.eight)]),
            BigTwoHand([Card(Suit.spades, Rank.nine)]),
            BigTwoHand([Card(Suit.spades, Rank.ten)]),
        ]
        env.state = [(3, [Card(Suit.spades, Rank.six)])]
        env.player_last_played = 3
        env.current_player = 0

        opponents_bots = [PlayFirstCardBot() for _ in range(3)]
        wrapped_env = SinglePlayerWrapper(env, 0, opponents_bots)

        # select the first card, spade seven
        raw_action = [1 if i == 0 else 0 for i in range(13)]

        action = PPOAction(
            transformed_obs=None,
            raw_action=raw_action,
            action_cat=None,
            action_mask=None,
            logp=None,
            cards=None,
        )

        _, reward, done = wrapped_env.step(action)
        self.assertTrue(done)
        self.assertEqual(reward, -1000)

    def test_step_continue_game(self):
        # how to do stateful object unit test the best practice??
        env = BigTwo()
        env.player_hands = [
            BigTwoHand([Card(Suit.spades, Rank.seven), Card(Suit.spades, Rank.ace)]),
            BigTwoHand([Card(Suit.spades, Rank.eight), Card(Suit.diamond, Rank.eight)]),
            BigTwoHand([Card(Suit.spades, Rank.nine), Card(Suit.diamond, Rank.nine)]),
            BigTwoHand([Card(Suit.spades, Rank.ten), Card(Suit.diamond, Rank.ten)]),
        ]
        env.state = [(3, [Card(Suit.spades, Rank.six)])]
        env.player_last_played = 3
        env.current_player = 0

        opponents_bots = [PlayFirstCardBot() for _ in range(3)]
        wrapped_env = SinglePlayerWrapper(env, 0, opponents_bots)

        # select the first card, spade seven
        raw_action = [1 if i == 0 else 0 for i in range(13)]

        action = PPOAction(
            transformed_obs=None,
            raw_action=raw_action,
            action_cat=None,
            action_mask=None,
            logp=None,
            cards=None,
        )

        _, reward, done = wrapped_env.step(action)
        self.assertFalse(done)
        self.assertEqual(reward, 0)
