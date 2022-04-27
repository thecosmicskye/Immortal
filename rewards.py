import numpy as np
from rlgym.utils.common_values import CAR_MAX_SPEED

from rlgym.utils import RewardFunction
from rlgym.utils import common_values
from rlgym.utils.gamestates import GameState, PlayerData

ball_max_height = common_values.CEILING_Z - common_values.BALL_RADIUS

class JumpTouchReward(RewardFunction):
    def __init__(self, min_height=common_values.BALL_RADIUS, exp=1):
        self.min_height = min_height
        self.exp = exp
        self.div = common_values.CEILING_Z ** self.exp
        self.ticks_until_next_reward = 0

    def reset(self, initial_state: GameState):
        self.ticks_until_next_reward = 0

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched \
                and not player.on_ground \
                and state.ball.position[2] >= self.min_height \
                and self.ticks_until_next_reward <= 0:
            self.ticks_until_next_reward = 47
            reward = (((state.ball.position[2] - common_values.BALL_RADIUS) ** self.exp) / self.div)
            if reward > .05:
                print(f"Aerial hit! % from ceiling: {round(reward*100,2)}%")
            return reward
        self.ticks_until_next_reward -= 1
        return 0

class WallTouchReward(RewardFunction):
    def __init__(self, min_height=common_values.BALL_RADIUS, exp=1):
        self.min_height = min_height
        self.exp = exp
        self.div = common_values.CEILING_Z ** self.exp

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched and player.on_ground and state.ball.position[2] >= self.min_height:
            reward = (((state.ball.position[2] - common_values.BALL_RADIUS) ** self.exp) / self.div)
            print(f"Wall hit! % from ceiling: {round(reward*100,2)}%")
            return reward

        return 0


class KickoffReward(RewardFunction):
    """
    a simple reward that encourages driving towards the ball fast while it's in the neutral kickoff position
    """
    def __init__(self):
        super().__init__()
        self.div = CAR_MAX_SPEED ** 2

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if state.ball.position[0] == 0 and state.ball.position[1] == 0:
            vel = player.car_data.linear_velocity
            pos_diff = state.ball.position - player.car_data.position
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            vel_to_ball = float(np.dot(norm_pos_diff, vel))
            vtb_exp = (vel_to_ball ** 2 / self.div) #* .5
            #print(f"KICKOFF: VTB: {vel_to_ball} Reward: {vtb_exp}")
            reward += vtb_exp
            #if player.boost_amount > 0:
                #boost_reward = (previous_action[6] > 0) * .5
                #print(f"KICKOFF: BOOSTY REWARD: {boost_reward}")
                #reward += boost_reward
        return reward