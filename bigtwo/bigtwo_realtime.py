from bigtwo import BigTwo
import numpy as np


class BigTwoRealTime(BigTwo):
    def __init__(self):
        super().__init__()

    def reset(self):
        super(BigTwoRealTime, self).reset()

        return [super(BigTwoRealTime, self)._current_observation(i) for i in range(4)]

    def step(self, action_n):
        current_player_number = super()._get_current_player()

        reward_n = []
        obs_n = []
        done_n = []
        for i in range(4):
            if current_player_number == i:
                obs, reward, done = super().step(action_n[i])
                obs_n.append(obs)
                reward_n.append(reward)
                done_n = [done for _ in range(4)]
            else:
                reward = -1 if np.count_nonzero(action_n[i]) > 0 else 0
                obs = super()._current_observation(i)
                obs_n.append(obs)
                reward_n.append(reward)

        return obs_n, reward_n, done_n
