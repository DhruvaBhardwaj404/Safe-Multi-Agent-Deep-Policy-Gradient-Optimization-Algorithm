import sys
import time

import numpy as np
import pandas as pd
from networkx.algorithms.structuralholes import constraint

from pettingzoo.mpe import simple_spread_v3
from src.MADDPG import MADDPG
from src.CMADDPG import CMADDPG
from src.helpers import *
import torch


def run_CMADDPG():
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    M = 4096
    length = 50
    c = np.array([0.5, 0.5, 0.5])
    env = simple_spread_v3.parallel_env(render_mode="human", max_cycles=length)

    control = CMADDPG(18, 5, 3, 0.95, 1.1, 0.1, 32, device, c)

    for episode in range(M):
        observations, infos = env.reset()

        for t in range(length):
            act = control.get_action(observations)
            act_ind = get_max_action_index(act)
            actions = {agent: action for agent, action in zip(env.agents, act_ind)}
            observations, rewards, terminations, truncations, infos, cost = env.step(actions)
            control.add_to_replay(observations, act, rewards, observations, cost)
            control.update()

        if episode % 10 == 0:
            control.save_results()

    env.close()

def run_MADDPG():
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    M = 10000
    length = 50
    c = np.array([0.5, 0.5, 0.5])
    env = simple_spread_v3.parallel_env(render_mode="human", max_cycles=length)

    control = MADDPG(18, 5, 3, 0.95, 1.1, 0.01, 10, device,batch_size=32)
    t1 = int(time.time())

    for episode in range(M):
        t2 = int(time.time())
        print(int(t2)-int(t1))
        t1 = time.time()

        observations, infos = env.reset()
        mean_reward = 0
        for t in range(length):
            act = control.get_action(observations)
            act_ind = get_max_action_index(act)
            actions = {agent: action for agent, action in zip(env.agents, act_ind)}
            observations, rewards, terminations, truncations, infos, cost = env.step(actions)
            temp = 0
            for k in rewards:
                rewards[k] = rewards[k] + cost[k]
                temp += rewards[k]
            mean_reward+=temp
            control.add_to_replay(observations, act, rewards, observations)
            control.update()
        mean_reward/=length
        control.performance_logs = pd.concat((control.performance_logs,pd.DataFrame({"episode":[episode],"mean reward":[mean_reward]})),ignore_index=True)

        if episode % 10 == 0:
            control.save_results()


    env.close()

if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == "M":
        run_MADDPG()
    elif mode == "C":
        run_CMADDPG()