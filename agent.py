from torch.nn import Sequential, Linear, LeakyReLU

from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.utils.util import SplitLayer


def get_critic(state_dim):
    return Sequential(
        Linear(state_dim, 512), LeakyReLU(),
        Linear(512, 512), LeakyReLU(),
        Linear(512, 512), LeakyReLU(),
        Linear(512, 512), LeakyReLU(),
        Linear(512, 512), LeakyReLU(),
        Linear(512, 512), LeakyReLU(),
        Linear(512, 1))


def get_actor(split, state_dim):
    total_output = sum(split)
    return DiscretePolicy(Sequential(Linear(state_dim, 512), LeakyReLU(),
                                     Linear(512, 512), LeakyReLU(),
                                     Linear(512, 512), LeakyReLU(),
                                     Linear(512, 512), LeakyReLU(),
                                     Linear(512, 512), LeakyReLU(),
                                     Linear(512, 512), LeakyReLU(),
                                     Linear(512, total_output),
                                     SplitLayer(splits=split)
                                     ), split)