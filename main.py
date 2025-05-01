import sys
import time
from copy import deepcopy

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from pettingzoo.mpe import simple_spread_v3
from src.MADDPG import MADDPG
from src.CMADDPG import CMADDPG
from src.CMADDPG_NoQ import CMADDPG_NQ
from src.helpers import *
import torch
from tqdm import tqdm
import imageio
from pympler import tracker
import gc

LOG_EVERY = 100
FLUSH_EVERY = 10000
TRAIN_EVERY = 100

MAX_EPISODES = 200000
EPISODE_LENGTH =25

VISUALIZE_EVERY = 10000
DISCOUNT_FACTOR = 0.95

BATCH_SIZE = 1024
TAU = 0.1


num_agents = 2
obs_shape = num_agents*6


def run_CMADDPG_with_Q_cost():
    torch.manual_seed(200)
    np.random.seed(200)
    device = "cpu"#("cuda" if torch.cuda.is_available() else "cpu")

    #c = np.array([0.3, 0.3, 0.3])
    c = np.array([0.01,0.01])
    env = simple_spread_v3.parallel_env(N=num_agents,render_mode="ansi", max_cycles=EPISODE_LENGTH)
    writer = SummaryWriter()
    control = CMADDPG(obs_shape, 5, num_agents, DISCOUNT_FACTOR,TAU, device, c,batch_size=BATCH_SIZE)
    epoch = 0
    discount_factors = [pow(DISCOUNT_FACTOR, i) for i in range(1, 25)]

    for episode in tqdm(range(MAX_EPISODES+1)):

        observations, infos = env.reset(episode)
        eps_distance = 0
        eps_reward = 0
        eps_cost = 0

        temp_cost = []
        temp_obs = []
        temp_nobs = []
        temp_act = []
        temp_rewards = []

        for t in range(EPISODE_LENGTH):
            epoch += 1
            act = control.get_action(observations)
            act_ind = get_max_action_index(act)
            actions = {agent: action for agent, action in zip(env.agents, act_ind)}
            nobservations, rewards, terminations, truncations, infos, cost = env.step(actions)

            temp_obs.append(observations)
            temp_nobs.append(nobservations)
            temp_act.append(act)
            temp_rewards.append(rewards)
            temp_cost.append(cost)

            distance = convert_dict_to_tensors(rewards)
            mean_cost = convert_dict_to_tensors(cost).mean()
            eps_reward += max(distance)
            eps_distance += -1 * distance.mean()
            eps_cost += mean_cost

            observations = nobservations

        for i in range(0,EPISODE_LENGTH):
            for j in range(i+1,EPISODE_LENGTH):
                for k in temp_cost[j].keys():
                    temp_cost[i][k]+=temp_cost[j][k]*discount_factors[j-i-1]
            for k in temp_cost[i].keys():
                temp_cost[i][k] /= (EPISODE_LENGTH-i)


        for observations, act, rewards, nobservations, cost in zip(temp_obs, temp_act, temp_rewards, temp_nobs,
                                                                   temp_cost):
            control.add_to_replay(observations, act, rewards, nobservations, cost)

        if episode % TRAIN_EVERY ==0:
            # memory_tracker.print_diff()
            Q_loss, C_loss, Dual_variable,J_c = control.update()
            if Q_loss is not None:
                writer.add_scalar("C loss", C_loss, epoch)
                writer.add_scalar("Q loss", Q_loss, epoch)
                writer.add_scalar("J_c Train", J_c, epoch)
                for i, l in enumerate(Dual_variable):
                    writer.add_scalar(f"Lambda {i}", l, epoch)
            # memory_tracker.print_diff()
            gc.collect()

        if episode % LOG_EVERY ==0:
            writer.add_scalar("Accumulated Global Reward", eps_reward, epoch)
            writer.add_scalar("Mean Distance From closest Landmark",eps_distance,epoch)
            writer.add_scalar("Total Cost Incurred during the episode", eps_cost, epoch)

            temp_J_C = [0 for _ in range(EPISODE_LENGTH)]

            for i, c in enumerate(temp_cost):
                for k in c:
                    temp_J_C[i] = c[k]
                temp_J_C[i] /= num_agents

            temp_J_C = np.mean(temp_J_C)

            writer.add_scalar("J_c", np.mean(temp_J_C), epoch)

        if episode % FLUSH_EVERY == 0:
            # control.save_results()
            writer.flush()

        if episode % VISUALIZE_EVERY == 0:
            l_env = simple_spread_v3.parallel_env(N=num_agents,render_mode="rgb_array", max_cycles=EPISODE_LENGTH, dynamic_rescaling=True)
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

def run_CMADDPG():
    torch.manual_seed(200)
    np.random.seed(200)
    device = "cpu"  # ("cuda" if torch.cuda.is_available() else "cpu")

    # c = np.array([0.3, 0.3, 0.3])
    c = np.array([0.01,0.01])
    env = simple_spread_v3.parallel_env(N=num_agents, render_mode="ansi", max_cycles=EPISODE_LENGTH)
    writer = SummaryWriter()

    control = CMADDPG_NQ(obs_shape, 5, num_agents, DISCOUNT_FACTOR, TAU, device, c, batch_size=BATCH_SIZE)
    epoch = 0

    discount_factors = [pow(DISCOUNT_FACTOR, i) for i in range(1, 25)]

    for episode in tqdm(range(MAX_EPISODES + 1)):

        observations, infos = env.reset(episode)
        eps_distance = 0
        eps_reward = 0
        eps_cost = 0

        temp_cost = []
        temp_obs = []
        temp_nobs = []
        temp_act = []
        temp_rewards = []

        for t in range(EPISODE_LENGTH):
            epoch += 1
            act = control.get_action(observations)
            act_ind = get_max_action_index(act)
            actions = {agent: action for agent, action in zip(env.agents, act_ind)}
            nobservations, rewards, terminations, truncations, infos, cost = env.step(actions)

            temp_obs.append(observations)
            temp_nobs.append(nobservations)
            temp_act.append(act)
            temp_rewards.append(rewards)
            temp_cost.append(cost)

            observations = nobservations

            distance = convert_dict_to_tensors(rewards)
            mean_cost = convert_dict_to_tensors(cost).mean()
            eps_reward += max(distance)
            eps_distance += -1 * distance.mean()
            eps_cost += mean_cost

        for i in range(0,EPISODE_LENGTH):
            for j in range(i+1,EPISODE_LENGTH):
                for k in temp_cost[j].keys():
                    temp_cost[i][k]+=temp_cost[j][k]*discount_factors[j-i-1]
            for k in temp_cost[i].keys():
                temp_cost[i][k] /= (EPISODE_LENGTH-i)

        for observations,act,rewards,nobservations,cost in zip(temp_obs,temp_act,temp_rewards,temp_nobs,temp_cost):
            control.add_to_replay(observations, act, rewards, nobservations, cost)

        if episode % TRAIN_EVERY == 0:
            # memory_tracker.print_diff()
            Q_loss, C_loss, Dual_variable, J_c = control.update()
            if Q_loss is not None:
                writer.add_scalar("Q loss", Q_loss, epoch)
                writer.add_scalar("J_c Train", J_c, epoch)
                for i, l in enumerate(Dual_variable):
                    writer.add_scalar(f"Lambda {i}", l, epoch)
            # memory_tracker.print_diff()
            gc.collect()

        if episode % LOG_EVERY == 0:
            writer.add_scalar("Accumulated Global Reward", eps_reward, epoch)
            writer.add_scalar("Mean Distance From closest Landmark", eps_distance,epoch)
            writer.add_scalar("Total Cost Incurred during the episode", eps_cost, epoch)


            temp_J_C = [0 for _ in range(EPISODE_LENGTH)]
            for i,c in enumerate(temp_cost):
                for k in c:
                    temp_J_C[i]=c[k]
                temp_J_C[i]/=num_agents

            temp_J_C = np.mean(temp_J_C)

            writer.add_scalar("J_c", np.mean(temp_J_C), epoch)

        if episode % FLUSH_EVERY == 0:
            # control.save_results()
            writer.flush()

        if episode % VISUALIZE_EVERY == 0:
            l_env = simple_spread_v3.parallel_env(N=num_agents, render_mode="rgb_array", max_cycles=EPISODE_LENGTH, dynamic_rescaling=True)
            for st in range(1, 6):
                observations, infos = l_env.reset(episode + st)
                frames = []

                for _ in range(EPISODE_LENGTH):
                    act = control.get_action(observations)
                    act_ind = get_max_action_index(act)
                    actions = {agent: action for agent, action in zip(l_env.agents, act_ind)}
                    observations, rewards, terminations, truncations, infos, cost = l_env.step(actions)
                    frame = l_env.render()
                    frame = np.array(frame)
                    frames.append(frame)

                imageio.mimsave(f'./Training_Visualizations/CMADDPG_{st}_{episode}.gif', frames, fps=5,
                                quantizer="mediancut")
            l_env.close()

    env.close()


def run_MADDPG():
    torch.manual_seed(200)
    np.random.seed(200)
    device = "cpu"#("cuda" if torch.cuda.is_available() else "cpu")

    env = simple_spread_v3.parallel_env(N=num_agents,render_mode="ansi", max_cycles=EPISODE_LENGTH)
    writer = SummaryWriter()

    discount_factors = [pow(DISCOUNT_FACTOR, i) for i in range(1, 25)]

    control = MADDPG(obs_shape, 5, num_agents, DISCOUNT_FACTOR, TAU, device,batch_size=BATCH_SIZE)
    t1 = int(time.time())
    epoch = 0
    for episode in tqdm(range(MAX_EPISODES+1)):

        observations, infos = env.reset(episode)
        eps_reward = 0
        mean_distance = 0
        eps_cost = 0

        temp_cost = []
        temp_obs = []
        temp_nobs = []
        temp_act = []
        temp_rewards = []

        for t in range(EPISODE_LENGTH):
            epoch += 1
            act = control.get_action(observations)
            act_ind = get_max_action_index(act)
            actions = {agent: action for agent, action in zip(env.agents, act_ind)}
            nobservations, rewards, terminations, truncations, infos, cost = env.step(actions)

            temp_obs.append(observations)
            temp_nobs.append(nobservations)
            temp_act.append(act)
            temp_rewards.append(rewards)
            temp_cost.append(cost)

            env_reward = deepcopy(rewards)
            for k in rewards:
                rewards[k] = rewards[k] #- cost[k]

            distance = convert_dict_to_tensors(env_reward)
            mean_cost = convert_dict_to_tensors(cost).mean()
            eps_reward += max(distance)
            mean_distance += -1*distance.mean()
            eps_cost += mean_cost


            observations=nobservations

        # control.performance_logs = pd.concat((control.performance_logs,pd.DataFrame({"episode":[episode],"mean reward":[mean_reward]})),ignore_index=True)
        for i in range(0, EPISODE_LENGTH):
            for j in range(i + 1, EPISODE_LENGTH):
                for k in temp_cost[j].keys():
                    temp_cost[i][k] += temp_cost[j][k] * discount_factors[j - i - 1]
            for k in temp_cost[i].keys():
                temp_cost[i][k] /= (EPISODE_LENGTH - i)

        for observations, act, rewards, nobservations, cost in zip(temp_obs, temp_act, temp_rewards, temp_nobs,
                                                                   temp_cost):
            control.add_to_replay(observations, act, rewards, nobservations, cost)

        if episode % TRAIN_EVERY==0:
            # memory_tracker.print_diff()
            loss_q, J_c = control.update()
            if loss_q is not None:
                writer.add_scalar("Q loss", loss_q, epoch)
                writer.add_scalar("J_c Train", J_c, epoch)
            # memory_tracker.print_diff()
            gc.collect()
            # memory_tracker.print_diff()
        if episode % LOG_EVERY == 0:
            writer.add_scalar("Accumulated Global Reward", eps_reward, epoch)
            writer.add_scalar("Mean Distance From closest Landmark", mean_distance, epoch)
            writer.add_scalar("Total Cost Incurred during the episode", eps_cost, epoch)

            temp_J_C = [0.0 for _ in range(EPISODE_LENGTH)]
            for i,c in enumerate(temp_cost):
                for k in c:
                    temp_J_C[i]=c[k]
                temp_J_C[i]/=num_agents

            temp_J_C = np.mean(temp_J_C)

            writer.add_scalar("J_c", np.mean(temp_J_C), epoch)

        if episode % FLUSH_EVERY == 0:
            writer.flush()
        #     control.save_results()

        if episode % VISUALIZE_EVERY == 0:
            l_env = simple_spread_v3.parallel_env(N=num_agents,render_mode="rgb_array", max_cycles=EPISODE_LENGTH, dynamic_rescaling=True)
            for st in range(1,6):
                observations, infos = l_env.reset(episode+st+234)
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

    memory_tracker = tracker.SummaryTracker()

    if algo == "M":
        run_MADDPG()
    elif algo == "C":
        run_CMADDPG_with_Q_cost()
    elif algo == "CNQ":
        run_CMADDPG()