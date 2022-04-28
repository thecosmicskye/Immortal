import argparse
import sys

import torch
from redis import Redis
from rlgym.envs import Match
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, \
    GoalScoredCondition

import learner
from learner import WORKER_COUNTER
from rocket_learn.agent.pretrained_agents.human_agent import HumanAgent
from rocket_learn.agent.pretrained_agents.necto.necto_v1 import NectoV1
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutWorker
from state import ImmortalStateSetter


def get_match(game_speed=100, human_match=False):

    frame_skip = 6  # Number of ticks to repeat an action
    fps = 120 / frame_skip

    terminals = [TimeoutCondition(round(fps * 30)),
                 NoTouchTimeoutCondition(round(fps * 20)),
                 GoalScoredCondition()],
    if human_match:
        terminals = [TimeoutCondition(round(fps * 180)),
                     GoalScoredCondition()],

    return Match(
        game_speed=game_speed,
        self_play=True,
        team_size=1,
        state_setter=ImmortalStateSetter(),
        obs_builder=learner.obs(),
        action_parser=learner.act(),
        terminal_conditions=[TimeoutCondition(round(fps * 30)),
                 NoTouchTimeoutCondition(round(fps * 20)),
                 GoalScoredCondition()],
        reward_function=learner.rew(),
        tick_skip=frame_skip,
    )


def make_worker(host, name, password, limit_threads=True, send_gamestates=False,
                is_streamer=False, human_match=False):
    if limit_threads:
        torch.set_num_threads(1)
    r = Redis(host=host, password=password)
    w = r.incr(WORKER_COUNTER) - 1

    model_name = "necto-model-30Y.pt"
    nectov1 = NectoV1(model_string=model_name, n_players=2)

    #EACH AGENT AND THEIR PROBABILITY OF OCCURRENCE
    agents = {nectov1: .2}

    human = None

    past_prob = .2
    eval_prob = .005
    game_speed = 100

    if is_streamer:
        past_prob = 0
        eval_prob = 0
        game_speed = 100

    if human_match:
        past_prob = 0
        eval_prob = 0
        game_speed = 1
        human = HumanAgent()


    return RedisRolloutWorker(r, name,
                              match=get_match(game_speed=game_speed,
                                              human_match=human_match),
                              past_version_prob=past_prob,
                              evaluation_prob=eval_prob,
                              send_gamestates=send_gamestates,
                              streamer_mode=False,
                              pretrained_agents=agents,
                              human_agent=human)


def main():
    assert len(sys.argv) >= 4

    parser = argparse.ArgumentParser(description='Launch worker')

    parser.add_argument('name', type=ascii,
                        help='<required> who is doing the work?')
    parser.add_argument('ip', type=ascii,
                        help='<required> learner ip')
    parser.add_argument('password', type=ascii,
                        help='<required> learner password')
    parser.add_argument('--compress', action='store_true',
                        help='compress sent data')
    parser.add_argument('--streamer_mode', action='store_true',
                        help='Start a streamer match, dont learn with this instance')
    parser.add_argument('--human_match', action='store_true',
                        help='Play a human match against Necto')

    args = parser.parse_args()

    name = args.name.replace("'", "")
    ip = args.ip.replace("'", "")
    password = args.password.replace("'", "")
    compress = args.compress
    stream_state = args.streamer_mode
    human_match = args.human_match

    try:
        worker = make_worker(ip, name, password,
                             limit_threads=True,
                             send_gamestates=compress,
                             is_streamer=stream_state,
                             human_match=human_match)
        worker.run()
    finally:
        print("Problem Detected. Killing Worker...")


if __name__ == '__main__':
    main()
