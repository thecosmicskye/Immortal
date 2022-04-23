import torch
import os

from agent import get_actor
from rocket_learn.agent.actor_critic_agent import ActorCriticAgent

split = (126,)

# TOTAL SIZE OF THE INPUT DATA
state_dim = 107

actor = get_actor(split, state_dim, True)

# PPO REQUIRES AN ACTOR/CRITIC AGENT

cur_dir = os.path.dirname(os.path.realpath(__file__))
checkpoint = torch.load(os.path.join(cur_dir, "checkpoint8.pt"))
actor.load_state_dict(checkpoint['actor_state_dict'])
actor.eval()
torch.jit.save(torch.jit.script(actor), 'jit.pt')