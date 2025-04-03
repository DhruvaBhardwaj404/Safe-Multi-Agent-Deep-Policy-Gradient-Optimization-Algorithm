import sys
import time
from copy import deepcopy

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from pettingzoo.mpe import simple_spread_v3
from src.MADDPG import MADDPG
from src.CMADDPG import CMADDPG
from src.helpers import *
import torch
from tqdm import tqdm

def run_CMADDPG(eps=0.5):

    device = "cpu"#("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # input()
    M = 100000
    length = 25
    c = np.array([0.5, 0.5, 0.5])
    env = simple_spread_v3.parallel_env(N=3,render_mode="ansi", max_cycles=length)
    writer = SummaryWriter()
    control = CMADDPG(18, 5, 3, 0.95, eps, 0.01, 10, device, c,batch_size=32)
    epoch = 0
    for episode in tqdm(range(M)):

        observations, infos = env.reset()

        for t in range(length):
            epoch += 1
            act = control.get_action(observations)
            act_ind = get_max_action_index(act)
            actions = {agent: action for agent, action in zip(env.agents, act_ind)}
            observations, rewards, terminations, truncations, infos, cost = env.step(actions)


            mean_reward = convert_dict_to_tensors(rewards).mean()
            mean_cost = convert_dict_to_tensors(cost).mean()

            writer.add_scalar("Reward", mean_reward, epoch)
            writer.add_scalar("Cost", mean_cost, epoch)

            control.add_to_replay(observations, act, rewards, observations, cost)
            Q_loss, C_loss, Dual_variable = control.update()
            if Q_loss is not None:
                writer.add_scalar("C loss", C_loss, epoch)
                writer.add_scalar("Q loss", Q_loss, epoch)
                for i,l in enumerate(Dual_variable):
                    writer.add_scalar(f"Lambda {i}", l, epoch)

        if episode % 10 == 0:
            # control.save_results()
            writer.flush()

    env.close()

def run_MADDPG(eps=0.5):
    device ="cpu"# ("cuda" if torch.cuda.is_available() else "cpu")
    M = 100000
    length = 25
    c = np.array([0.5, 0.5, 0.5])
    env = simple_spread_v3.parallel_env(N=3,render_mode="ansi", max_cycles=length)
    writer = SummaryWriter()
    control = MADDPG(18, 5, 3, 0.95, eps, 0.01, 10, device,batch_size=32,eps_decay=1000)
    t1 = int(time.time())
    epoch = 0
    for episode in tqdm(range(M)):
        # t2 = int(time.time())
        # # print(int(t2)-int(t1))
        # t1 = time.time()

        observations, infos = env.reset()
        mean_reward = 0
        for t in range(length):
            epoch += 1
            act = control.get_action(observations)
            act_ind = get_max_action_index(act)
            actions = {agent: action for agent, action in zip(env.agents, act_ind)}
            observations, rewards, terminations, truncations, infos, cost = env.step(actions)
            temp = 0
            env_reward = deepcopy(rewards)
            for k in rewards:
                rewards[k] = rewards[k] - cost[k]
                temp += rewards[k]
            mean_reward+=temp
            mean_env_reward = convert_dict_to_tensors(env_reward).mean()
            mean_modified_reward = convert_dict_to_tensors(rewards).mean()
            mean_cost = convert_dict_to_tensors(cost).mean()
            writer.add_scalar("Reward", mean_env_reward, epoch)
            writer.add_scalar("Modified Reward", mean_modified_reward, epoch)
            writer.add_scalar("Cost", mean_cost, epoch)

            control.add_to_replay(observations, act, rewards, observations)
            loss_q = control.update()
            if loss_q is not None:
                writer.add_scalar("Q loss", loss_q, epoch)
        mean_reward/= length
        # control.performance_logs = pd.concat((control.performance_logs,pd.DataFrame({"episode":[episode],"mean reward":[mean_reward]})),ignore_index=True)

        if episode % 10 == 0:
            writer.flush()
        #     control.save_results()


    env.close()

if __name__ == "__main__":
    algo = sys.argv[1]

    if algo == "M":
        run_MADDPG()
    elif algo == "C":
        run_CMADDPG()