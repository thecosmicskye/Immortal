from typing import Any

import gym.spaces
import numpy as np
from gym.spaces import Discrete
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.gamestates import GameState


class ImmortalAction(ActionParser):
    def __init__(self):
        super().__init__()
        self._lookup_table = self._make_lookup_table()

    @staticmethod
    def _make_lookup_table():
        actions = []
        # Ground
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake better car control
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, 1])
        actions = np.array(actions)
        return actions

    def get_action_space(self) -> gym.spaces.Space:
        return Discrete(len(self._lookup_table))

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        return self._lookup_table[actions.astype(int).squeeze()]


def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    intersected = np.setdiff1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])

SetAction = ImmortalAction

if __name__ == '__main__':
    ap = ImmortalAction()
    action_space = ap.get_action_space()
    table2 = ap._lookup_table
    print(ap.get_action_space())