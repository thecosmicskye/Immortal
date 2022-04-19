import os
import sys

import numpy as np
import torch.jit
from redis import Redis
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import VelocityReward, VelocityPlayerToBallReward, \
    VelocityBallToGoalReward, EventReward
from rlgym_tools.extra_rewards.kickoff_reward import KickoffReward

import wandb
from actionparser import ImmortalAction
from agent import get_critic, get_actor
from obs import ExpandAdvancedObs
from rewards import JumpTouchReward, WallTouchReward
from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator

WORKER_COUNTER = "worker-counter"


# ENSURE OBSERVATION, REWARD, AND ACTION CHOICES ARE THE SAME IN THE WORKER
def obs():
    return ExpandAdvancedObs()


def rew():
    return CombinedReward.from_zipped(
        (VelocityPlayerToBallReward(), 0.004),
        (VelocityReward(), 0.024),
        (VelocityBallToGoalReward(), 0.02),
        (KickoffReward(), 0.2),
        (JumpTouchReward(), 6.0),
        (WallTouchReward(min_height=250), 6.0),
        (EventReward(team_goal=1200,
                     save=200,
                     demo=500,
                     concede=-1000), 0.01),
    )


def act():
    return ImmortalAction()


def get_latest_checkpoint():
    subdir = 'checkpoint_save_directory'

    all_subdirs = [os.path.join(subdir, d) for d in os.listdir(subdir) if os.path.isdir(os.path.join(subdir, d))]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    all_subdirs = [os.path.join(latest_subdir, d) for d in os.listdir(latest_subdir) if
                   os.path.isdir(os.path.join(latest_subdir, d))]
    latest_subdir = (max(all_subdirs, key=os.path.getmtime))
    full_dir = os.path.join(latest_subdir, 'checkpoint.pt')
    print(full_dir)

    return full_dir


if __name__ == "__main__":
    """
    
    Starts up a rocket-learn learner process, which ingests incoming data, updates parameters
    based on results, and sends updated model parameters out to the workers
    
    """

    frame_skip = 6  # Number of ticks to repeat an action
    half_life_seconds = 15  # Easier to conceptualize, after this many seconds the reward discount is 0.5
    run_name = "Second"
    run_id = "2emtr6mw"
    #file = None
    file = get_latest_checkpoint()
    #file = "checkpoint_save_directory/Immortal_1650212001.201177/Immortal_4790/checkpoint.pt"

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))

    # LINK TO THE REDIS SERVER YOU SHOULD HAVE RUNNING (USE THE SAME PASSWORD YOU SET IN THE REDIS
    # CONFIG)
    _, ip, password, clear = sys.argv
    clear = clear.lower()

    if clear == 'true':
        print('clearing DB')
        clear = True
    else:
        print('not clearing DB')
        clear = False
    redis = Redis(host=ip, password=password)
    redis.delete(WORKER_COUNTER)  # Reset to 0

    config = dict(
        seed=125,
        actor_lr=2e-4,
        critic_lr=2e-4,
        ent_coef=0.01,
        n_steps=2_000_000,
        batch_size=300_000,
        minibatch_size=150_000,
        epochs=32,
        gamma=gamma,
        iterations_per_save=5
    )

    # ROCKET-LEARN USES WANDB WHICH REQUIRES A LOGIN TO USE. YOU CAN SET AN ENVIRONMENTAL VARIABLE
    # OR HARDCODE IT IF YOU ARE NOT SHARING YOUR SOURCE FILES
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(project="Immortal", entity=os.environ["entity"], id=run_id, config=config,
                        settings=wandb.Settings(_disable_stats=True))
    torch.manual_seed(logger.config.seed)

    # THE ROLLOUT GENERATOR CAPTURES INCOMING DATA THROUGH REDIS AND PASSES IT TO THE LEARNER.
    # -save_every SPECIFIES HOW OFTEN OLD VERSIONS ARE SAVED TO REDIS. THESE ARE USED FOR TRUESKILL
    # COMPARISON AND TRAINING AGAINST PREVIOUS VERSIONS
    rollout_gen = RedisRolloutGenerator(redis, obs, rew, act,
                                        logger=logger,
                                        save_every=logger.config.iterations_per_save,
                                        max_age=1, clear=clear)

    # ROCKET-LEARN EXPECTS A SET OF DISTRIBUTIONS FOR EACH ACTION FROM THE NETWORK, NOT
    # THE ACTIONS THEMSELVES. SEE network_setup.readme.txt FOR MORE INFORMATION
    split = (126,)

    # TOTAL SIZE OF THE INPUT DATA
    state_dim = 107

    critic = get_critic(state_dim)

    actor = get_actor(split, state_dim)

    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": logger.config.actor_lr},
        {"params": critic.parameters(), "lr": logger.config.critic_lr}
    ])

    # PPO REQUIRES AN ACTOR/CRITIC AGENT
    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)

    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=logger.config.ent_coef,
        n_steps=logger.config.n_steps,
        batch_size=logger.config.batch_size,
        minibatch_size=logger.config.minibatch_size,
        epochs=logger.config.epochs,
        gamma=logger.config.gamma,
        logger=logger,
    )

    # BEGIN TRAINING. IT WILL CONTINUE UNTIL MANUALLY STOPPED
    # -iterations_per_save SPECIFIES HOW OFTEN CHECKPOINTS ARE SAVED
    # -save_dir SPECIFIES WHERE

    if file:
        print(f'loading from {file}')
        alg.load(file, continue_iterations=True)

    alg.run(iterations_per_save=logger.config.iterations_per_save, save_dir="checkpoint_save_directory")
