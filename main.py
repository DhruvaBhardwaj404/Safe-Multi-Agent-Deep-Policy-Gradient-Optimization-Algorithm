from pettingzoo.mpe import simple_spread_v3
from src.MADDPG import MADDPG
from src.helpers import *
import torch

if __name__ == "__main__":
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    M = 4096
    length = 50
    env = simple_spread_v3.parallel_env(render_mode="human", max_cycles=length)
    control = MADDPG(18, 5, 3, 0.95, 1.1, 0.1, 32, device)


    for episode in range(M):
        observations, infos = env.reset()

        for t in range(length):
            act = control.get_action(observations)
            act_ind = get_max_action_index(act)
            actions = {agent: action for agent,action in zip(env.agents,act_ind)}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            control.add_to_replay(observations, act, rewards, observations)
            control.update()
    env.close()