from typing import Any

import numpy
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs


# ROCKET-LEARN ALWAYS EXPECTS A BATCH DIMENSION IN THE BUILT OBSERVATION
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)