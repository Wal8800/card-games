import typing
from typing import List, Tuple

import gym
import numpy as np
from gym.spaces import Box, Discrete
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from bigtwo.bigtwo import BigTwo, BigTwoHand
from bigtwo.preprocessing import (
    create_action_cat_mapping,
    generate_action_mask,
    obs_to_ohe,
)


class BigTwoMultiAgentEnv(MultiAgentEnv):
    def __init__(self, hands: typing.Optional[List[BigTwoHand]] = None):
        super().__init__()
        self.base_env = BigTwo(hands)

        self.cat_to_raw_action, self.raw_action_idx_to_cat = create_action_cat_mapping()

        max_avail_actions = len(self.cat_to_raw_action)

        init_obs = BigTwo().get_current_player_obs()

        ohe_obs = obs_to_ohe(init_obs)

        self.action_space = Discrete(max_avail_actions)
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": Box(0, 1, shape=(max_avail_actions,)),
                "game_obs": Box(
                    np.amin(ohe_obs), np.amax(ohe_obs), shape=ohe_obs.shape
                ),
            }
        )

        self.number_of_player = 4

        self.player_id_mapping = {
            i: f"player_{i}" for i in range(self.number_of_player)
        }

    def hands_played(self) -> int:
        return len(self.base_env.state)

    def actions_attempted(self) -> int:
        return len(self.base_env.past_actions)

    def step(
        self, action_dict: MultiAgentDict, with_raw_obs=False
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        curr_player = self.base_env.current_player

        player_id = self.player_id_mapping.get(curr_player)

        action_cat = action_dict[player_id]
        raw_action = self.cat_to_raw_action[action_cat]

        _, is_game_finished = self.base_env.step(raw_action)

        done = {
            "__all__": is_game_finished,
            **{
                player_id: is_game_finished
                for _, player_id in self.player_id_mapping.items()
            },
        }

        if not is_game_finished:
            rewards = {player_id: 0 for _, player_id in self.player_id_mapping.items()}
        else:
            rewards = {
                player_id: 1 if curr_player == idx else -1
                for idx, player_id in self.player_id_mapping.items()
            }

        obs = {}
        if not is_game_finished:
            next_player = self.base_env.current_player
            player_obs = self.base_env.get_current_player_obs()
            ohe_obs = obs_to_ohe(player_obs)
            obs = {
                self.player_id_mapping.get(next_player): {
                    "action_mask": generate_action_mask(
                        self.raw_action_idx_to_cat, player_obs
                    ),
                    "game_obs": ohe_obs,
                }
            }

            if with_raw_obs:
                obs[self.player_id_mapping.get(next_player)]["raw_obs"] = player_obs

        return obs, rewards, done, {}

    def reset(self, with_raw_obs=False) -> MultiAgentDict:
        obs = self.base_env.reset()

        player_id = self.player_id_mapping.get(obs.current_player)
        player_obs = self.base_env.get_current_player_obs()
        ohe_obs = obs_to_ohe(player_obs)

        obs = {
            player_id: {
                "action_mask": generate_action_mask(
                    self.raw_action_idx_to_cat, player_obs
                ),
                "game_obs": ohe_obs,
            }
        }

        if with_raw_obs:
            obs[player_id]["raw_obs"] = player_obs

        return obs


if __name__ == "__main__":
    new_env = BigTwoMultiAgentEnv()

    new_obs = new_env.reset()
    temp = 0
