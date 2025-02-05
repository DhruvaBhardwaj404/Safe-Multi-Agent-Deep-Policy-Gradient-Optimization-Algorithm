from copy import deepcopy
import torch
from agilerl.components.replay_buffer import ReplayBuffer
import numpy
import time
from src.helpers import *
from numpy.core.fromnumeric import argmax
from torchviz import make_dot
from src.Agent import Agent
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, ListStorage
import warnings


class MADDPG:
    """
    Implements Multi-Agent Deep Deterministic Policy Gradient Algorithm
    """
    def __init__(self,obs_size:int,action_size:int,agent_num:int,gamma:float, eps:float,tau:float,threshold:int,device:str, eps_decay:int=1000):
        """

        @param obs_size:
        @type obs_size:
        @param action_size:
        @type action_size:
        @param agent_num:
        @type agent_num:
        @param gamma:
        @type gamma:
        @param eps:
        @type eps:
        @param tau:
        @type tau:
        @param threshold:
        @type threshold:
        @param device:
        @type device:
        @param eps_decay:
        @type eps_decay:
        """
        torch.autograd.set_detect_anomaly(True)
        warnings.filterwarnings("ignore")
        self.action_size = action_size
        self.obs_size = obs_size
        self.agent_num = agent_num
        self.gamma = torch.tensor(gamma).to(device)
        self.eps = torch.tensor(eps).to(device)
        self.eps_decay = torch.tensor(eps_decay).to(device)
        self.tau = torch.tensor(tau).to(device)
        self.batch_size = 8
        self.replay = ReplayBuffer(storage=ListStorage(max_size=1024),batch_size=self.batch_size)
        self.q_optim = []
        self.p_optim = []
        self.threshold = threshold
        self.device = device
        self.steps = 0
        self.agents = []

        for i in range(0,agent_num):
            self.agents.append(Agent(i,action_size,obs_size,agent_num,device))
            self.agents[i].load_checkpoint()



    def add_to_replay(self,obs_old:torch.tensor, action:torch.tensor, reward:torch.tensor,
                      obs_new:torch.tensor):
        """

        @param obs_old:
        @type obs_old:
        @param action:
        @type action:
        @param reward:
        @type reward:
        @param obs_new:
        @type obs_new:
        @return:
        @rtype:
        """
        reward = convert_dict_to_tensors(reward).to("cpu")
        obs_old = convert_dict_to_tensors(obs_old).to("cpu")
        obs_new = convert_dict_to_tensors(obs_new).to("cpu")
        act = []
        for a in action:
            act.append(a.cpu())

        if reward.size()[0]==0:
            return
        self.replay.add(TensorDict({"obs":obs_old, "act":act, "rew":reward, "nobs":obs_new}))

    def get_action(self,obs):
        """

        @param obs:
        @type obs:
        @return:
        @rtype:
        """
        obs = convert_dict_to_tensors(obs).to(self.device)
        agent_actions = []

        sample = numpy.random.random()
        eps_threshold = self.eps + (self.eps - 0.05 ) * \
                        torch.exp(-1. * self.steps/ self.eps_decay)
        self.steps += 1
        for ind,a in enumerate(self.agents):
            if sample < eps_threshold:
                agent_actions.append(a.get_next_action(obs[ind],True))
            else:
                agent_actions.append(a.get_next_action(obs[ind]).to("cpu"))

        return agent_actions


    def update(self):
        """

        @return:
        @rtype:
        """
        if len(self.replay) < self.threshold:
            return

        samples = self.replay.sample()
        gnobs = [[] for _ in samples]
        gnact = [[] for _ in samples]

        gact_curr = [[] for _ in samples]
        act_curr = [[] for _ in self.agents]
        gobs = [[] for _ in samples]
        gact = [[] for _ in samples]
        rew = [0 for _ in range(self.batch_size)]
        nact = [[] for _ in self.agents]
        nobs = [[] for _ in self.agents]
        obs = [[] for _ in self.agents]

        for j,s in enumerate(samples):
            sample = s.to(self.device)
            for i,agent in enumerate(self.agents):
                rew[j] += sample["rew"][i]
                obs[i].append(sample["obs"][i])
                nobs[i].append(sample["nobs"][i])
                act_curr[i].append(sample["act"][i])
                nact[i].append(agent.get_action_target(nobs[i][-1]))
                gnobs[j].extend(sample["nobs"][i])
                gnact[j].extend(nact[i][-1])
                gact_curr[j].extend(act_curr[i][-1])
                gobs[j].extend(obs[i][-1])
                gact[j].extend(sample["act"][i])

        for i,agent in enumerate(self.agents):
            loss_q = torch.nn.MSELoss()
            q = []
            y = []
            for j in range(self.batch_size):
                qt_inp = torch.tensor(gnobs[i] + gnact[i]).to(self.device)
                q_inp = torch.tensor(gobs[i] + gact[i]).to(self.device)
                temp = torch.add(rew[j], self.gamma*agent.q_function_target(qt_inp))
                y.append(temp)
                temp = agent.get_reward(q_inp)
                q.append(temp)

            agent.q_grad.zero_grad()

            q= torch.cat(q).to(self.device)
            y= torch.cat(y).to(self.device)
            loss = loss_q(q,y)
            loss.backward()
            agent.q_grad.step()

            # if i == 0:
            #     for name,param in agent.q_function.model.named_parameters():
            #         print(name,param.grad)
            #     # input()
            #     # print(agent.q_function.model.state_dict())
            #     # input()

            exp_ret = torch.tensor(0,dtype=torch.float32)
            for j in range(self.batch_size):
                q_inp = torch.tensor(gobs[j]+gact_curr[j]).to(self.device)
                cur_pol = torch.tensor(act_curr[i][j]).to(self.device)
                cur_pol_weight = torch.nn.functional.gumbel_softmax(cur_pol,hard=True)
                cur_pol  = torch.matmul(cur_pol,cur_pol_weight)
                cur_rew = agent.get_reward(q_inp)
                exp_ret = exp_ret - cur_pol*cur_rew
            exp_ret = torch.divide(exp_ret, self.batch_size)
            agent.policy_grad.zero_grad()
            exp_ret.backward()
            agent.policy_grad.step()

            # if i == 0:
            #     for name,param in agent.policy.model.named_parameters():
            #         print(name,param.grad)
            #         input()
            #         print(agent.policy.model.state_dict())
            #         input()

            for target_param, param in zip(agent.policy_target.model.parameters(), agent.policy.model.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

            for target_param, param in zip(agent.q_function_target.model.parameters(), agent.q_function.model.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

            agent.save_checkpoint(loss,exp_ret)