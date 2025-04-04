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
import imageio


LOG_EVERY = 100
FLUSH_EVERY = 10000
TRAIN_EVERY = 100

MAX_EPISODES = 250000
EPISODE_LENGTH =25

VISUALIZE_EVERY = 1000

num_agents = 2
obs_shape = num_agents*6


def run_CMADDPG():
    torch.manual_seed(100)
    np.random.seed(100)
    device = "cpu"#("cuda" if torch.cuda.is_available() else "cpu")

    c = np.array([0.3, 0.3, 0.3])

    env = simple_spread_v3.parallel_env(N=num_agents,render_mode="ansi", max_cycles=EPISODE_LENGTH)
    writer = SummaryWriter()
    control = CMADDPG(obs_shape, 5, num_agents, 0.95,0.01, device, c,batch_size=1024)
    epoch = 0
    for episode in tqdm(range(MAX_EPISODES)):

        observations, infos = env.reset(episode)
        eps_rew = 0
        eps_cost = 0
        for t in range(EPISODE_LENGTH):
            epoch += 1
            act = control.get_action(observations)
            act_ind = get_max_action_index(act)
            actions = {agent: action for agent, action in zip(env.agents, act_ind)}
            observations, rewards, terminations, truncations, infos, cost = env.step(actions)


            mean_reward = convert_dict_to_tensors(rewards).mean()
            mean_cost = convert_dict_to_tensors(cost).mean()

            eps_rew+=mean_reward
            eps_cost+=mean_cost
            control.add_to_replay(observations, act, rewards, observations, cost)


        if episode % TRAIN_EVERY ==0:
            Q_loss, C_loss, Dual_variable = control.update()
            if Q_loss is not None:
                writer.add_scalar("C loss", C_loss, epoch)
                writer.add_scalar("Q loss", Q_loss, epoch)
                for i, l in enumerate(Dual_variable):
                    writer.add_scalar(f"Lambda {i}", l, epoch)

        if episode % LOG_EVERY ==0:
            writer.add_scalar("Reward", eps_rew, epoch)
            writer.add_scalar("Cost", eps_cost, epoch)

        if episode % FLUSH_EVERY == 0:
            # control.save_results()
            writer.flush()

        if episode % VISUALIZE_EVERY == 0:
            l_env = simple_spread_v3.parallel_env(N=num_agents,render_mode="rgb_array", max_cycles=EPISODE_LENGTH)
            for st in range(1,6):
                observations, infos = l_env.reset(episode+st)
                frames = []

                for _ in range(EPISODE_LENGTH):
                    act = control.get_action(observations)
                    act_ind = get_max_action_index(act)
                    actions = {agent: action for agent, action in zip(l_env.agents, act_ind)}
                    observations, rewards, terminations, truncations, infos, cost = l_env.step(actions)
                    frame = l_env.render()
                    frame = np.array(frame)
                    frames.append(frame)

                imageio.mimsave(f'./Training_Visualizations/CMADDPG_{st}_{episode}.gif', frames, fps=5,quantizer="mediancut")
            l_env.close()

    env.close()

def run_MADDPG():
    torch.manual_seed(100)
    np.random.seed(100)
    device = "cpu" #("cuda" if torch.cuda.is_available() else "cpu")

    env = simple_spread_v3.parallel_env(N=num_agents,render_mode="ansi", max_cycles=EPISODE_LENGTH)
    writer = SummaryWriter()
    control = MADDPG(obs_shape, 5, num_agents, 0.95, 0.01, device,batch_size=1024)
    t1 = int(time.time())
    epoch = 0
    for episode in tqdm(range(MAX_EPISODES)):

        observations, infos = env.reset(episode)
        eps_reward = 0
        eps_env_reward = 0
        eps_cost = 0

        for t in range(EPISODE_LENGTH):
            epoch += 1
            act = control.get_action(observations)
            act_ind = get_max_action_index(act)
            actions = {agent: action for agent, action in zip(env.agents, act_ind)}
            observations, rewards, terminations, truncations, infos, cost = env.step(actions)

            env_reward = deepcopy(rewards)
            for k in rewards:
                rewards[k] = rewards[k] - cost[k]

            mean_env_reward = convert_dict_to_tensors(env_reward).mean()
            mean_reward = convert_dict_to_tensors(rewards).mean()
            mean_cost = convert_dict_to_tensors(cost).mean()
            eps_reward += mean_reward
            eps_env_reward += mean_env_reward
            eps_cost += mean_cost

            control.add_to_replay(observations, act, rewards, observations)
        # control.performance_logs = pd.concat((control.performance_logs,pd.DataFrame({"episode":[episode],"mean reward":[mean_reward]})),ignore_index=True)

        if episode % TRAIN_EVERY==0:
            loss_q = control.update()
            if loss_q is not None:
                writer.add_scalar("Q loss", loss_q, epoch)

        if episode % LOG_EVERY == 0:
            writer.add_scalar("Algorithm Reward", eps_reward, epoch)
            writer.add_scalar("Reward", eps_env_reward, epoch)
            writer.add_scalar("Cost", eps_cost, epoch)

        if episode % FLUSH_EVERY == 0:
            writer.flush()
        #     control.save_results()

        if episode % VISUALIZE_EVERY == 0:
            l_env = simple_spread_v3.parallel_env(N=num_agents,render_mode="rgb_array", max_cycles=EPISODE_LENGTH)
            for st in range(1,6):
                observations, infos = l_env.reset(episode+st)
                frames = []

                for _ in range(EPISODE_LENGTH):
                    act = control.get_action(observations)
                    act_ind = get_max_action_index(act)
                    actions = {agent: action for agent, action in zip(l_env.agents, act_ind)}
                    observations, rewards, terminations, truncations, infos, cost = l_env.step(actions)
                    frame = l_env.render()
                    frame = np.array(frame)
                    frames.append(frame)

                imageio.mimsave(f'./Training_Visualizations/CMADDPG_{st}_{episode}.gif', frames, fps=5,quantizer="mediancut")
            l_env.close()

    env.close()

if __name__ == "__main__":
    algo = sys.argv[1]

    if algo == "M":
        run_MADDPG()
    elif algo == "C":
        run_CMADDPG()