from copy import deepcopy

import numpy as np
import pandas as pd
import torch.nn
from agilerl.components.replay_buffer import ReplayBuffer
import numpy
import time
from src.helpers import *
from src.Agent import ConstrainedAgent as Agent
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, ListStorage
import warnings


class CMADDPG:
    """
    Implements Multi-Agent Deep Deterministic Policy Gradient Algorithm
    """
    def __init__(self,obs_size:int,action_size:int,agent_num:int,gamma:float,
                 tau:float,device:str,local_constraints, batch_size = 1024):
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
        # self.eps = torch.tensor(eps).to(device)
        self.tau = torch.tensor(tau).to(device)
        self.batch_size = batch_size
        self.replay = ReplayBuffer(storage=ListStorage(max_size=1000000),batch_size=self.batch_size)
        self.q_optim = []
        self.p_optim = []
        # self.threshold = threshold
        self.device = device
        self.steps = 0
        self.agents = []
        self.dual_variable = [torch.tensor(5.0, requires_grad=True) for c in local_constraints]
        self.dual_optim = torch.optim.Adam(self.dual_variable,lr=0.001)

        self.local_constraints = local_constraints
        # try:
        #     self.training_logs = pd.read_csv("./training_record.csv")
        # except Exception as e:
        #     self.training_logs = pd.DataFrame()

        for i in range(0,agent_num):
            self.agents.append(Agent(i,action_size,obs_size,agent_num,device))
            self.agents[i].load_checkpoint()



    def add_to_replay(self,obs_old:torch.tensor, action:torch.tensor, reward:torch.tensor,
                      obs_new:torch.tensor, cost:torch.tensor):
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
        cost = convert_dict_to_tensors(cost).to("cpu")
        obs_old = convert_dict_to_tensors(obs_old).to("cpu")
        obs_new = convert_dict_to_tensors(obs_new).to("cpu")
        act = torch.stack(action) #convert_dict_to_tensors(a)
        # for a in action:
        #     act.append(a)
        # print(act)
        # act = torch.tensor([act])
        self.replay.add({"obs":obs_old, "act":act, "rew":reward, "nobs":obs_new, "cost":cost})

    def get_action(self,obs):
        """

        @param obs:
        @type obs:
        @return:
        @rtype:
        """
        obs = convert_dict_to_tensors(obs).to(self.device)
        agent_actions = []

        for ind,a in enumerate(self.agents):
            agent_actions.append(a.get_next_action(obs[ind]).to("cpu"))

        return agent_actions


    def update(self):
        """

        @return:
        @rtype:
        """
        if len(self.replay) < self.batch_size:
            return None,None,None

        num_agents = len(self.agents)
        samples = self.replay.sample()

        rew_batch = samples["rew"].to(self.device).transpose(0, 1).clone().detach()
        cost_batch = samples["cost"].to(self.device).transpose(0, 1).clone().detach()
        obs_batch = samples["obs"].to(self.device).transpose(0, 1).clone().detach()
        nobs_batch = samples["nobs"].to(self.device).transpose(0, 1).clone().detach()
        act_batch = samples["act"].to(self.device).transpose(0, 1).clone()

        nact_batch = torch.stack([
            self.agents[i].get_action_target(nobs_batch[i])
            for i in range(num_agents)
        ])
        cur_act_batch = torch.stack([
            self.agents[i].get_next_action(obs_batch[i])
            for i in range(num_agents)
            ])

        gobs_batch = obs_batch.permute(1, 0, 2).reshape(self.batch_size, self.obs_size*num_agents)
        gact_batch = act_batch.permute(1, 0, 2).reshape(self.batch_size, self.action_size*num_agents)
        gnobs_batch = nobs_batch.permute(1, 0, 2).reshape(self.batch_size, self.obs_size*num_agents)
        gnact_batch = nact_batch.permute(1, 0, 2).reshape(self.batch_size, self.action_size*num_agents)
        gcur_act_batch = cur_act_batch.permute(1, 0, 2).reshape(self.batch_size, self.action_size*num_agents)

        q_target_input = torch.cat([gnobs_batch, gnact_batch], dim=1)
        q_input = torch.cat([gobs_batch, gact_batch], dim=1)
        q_input_pol = torch.cat([gobs_batch, gcur_act_batch], dim=1)

        mean_q_loss_reward = 0
        mean_q_loss_cost = 0

        for i, agent in enumerate(self.agents):

            q_input_r = q_input.clone().detach()
            q_input_t_r = q_target_input.clone().detach()
            rew = rew_batch[i].view((self.batch_size,1)).clone().detach()
            q_r_t = agent.q_function_target_r(q_input_t_r)
            y_target_r = rew + self.gamma * q_r_t

            q_pred_r = agent.get_reward(q_input_r)

            loss_q_r = torch.nn.MSELoss()(q_pred_r, y_target_r)
            with torch.no_grad():
                mean_q_loss_reward += loss_q_r
            agent.q_grad_r.zero_grad()
            loss_q_r.backward(retain_graph=True)
            agent.q_grad_r.step()
            cost = cost_batch[i].view((self.batch_size,1)).clone().detach()
            q_input_c = q_input.clone().detach()
            q_input_t_c = q_target_input.clone().detach()

            q_c_t = agent.q_function_target_c(q_input_t_c)
            y_target_c = cost + self.gamma * q_c_t

            q_pred_c = agent.get_cost(q_input_c)

            loss_q_c = torch.nn.MSELoss()(q_pred_c, y_target_c)
            with torch.no_grad():
                mean_q_loss_cost += loss_q_c
            agent.q_grad_c.zero_grad()
            loss_q_c.backward(retain_graph=True)
            agent.q_grad_c.step()

            q_input_p = q_input_pol.clone().detach()

            cur_policy = agent.get_next_action(obs_batch[i])

            pol_weight = torch.nn.functional.gumbel_softmax(cur_policy, hard=True)

            cur_pol = (cur_policy * pol_weight).sum(dim=1).clone()
            cur_pol = cur_pol.view((self.batch_size,1))
            log_pol = torch.log(cur_pol)

            q_value = agent.get_reward(q_input_p)

            exp_ret =  - (log_pol * q_value)
            exp_ret = exp_ret + self.dual_variable[i] * (log_pol * cost - self.local_constraints[i])
            exp_ret = exp_ret.mean()

            agent.policy_grad.zero_grad()
            self.dual_optim.zero_grad()

            exp_ret.backward(retain_graph=True)

            agent.policy_grad.step()
            self.dual_optim.step()



            with torch.no_grad():
                for target_param, param in zip(agent.policy_target.model.parameters(), agent.policy.model.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

                for target_param, param in zip(agent.q_function_target_c.model.parameters(), agent.q_function_c.model.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

                for target_param, param in zip(agent.q_function_target_r.model.parameters(), agent.q_function_r.model.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

                # agent.save_checkpoint(loss_q_r,loss_q_c,exp_ret)

        return mean_q_loss_reward, mean_q_loss_cost, self.dual_variable
            # self.training_logs = pd.concat((self.training_logs, pd.DataFrame({"timestamp":int(time.time()),"agent_num":i,"mean_q_loss":loss.detach().to("cpu"),"mean_exp_return":exp_ret.detach().to("cpu")})),ignore_index=True)

    # def save_results(self):
    #     self.training_logs.to_csv("./training_record_CMADDPG.csv")