from copy import deepcopy
import torch
import numpy.random
from torch import nn
from src.DPN import DPN
from src.DQN import DQN


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

class Agent:
    """
    Contains methods, data and models that is related to agents (such as critic model etc.)
    """

    def __init__(self, agent_id:int, action_size:int, obs_size:int, num_agents:int,device:str="cpu",lr=0.01):
        """
        @param agent_id: unique identifier for the agent
        @type agent_id: integer
        @param action_size: size of action space
        @type action_size: integer
        @param obs_size: size of observation space
        @type obs_size: integer
        @param num_agents: number of independent agent in the environment
        @type num_agents: integer
        @param device: device memory to use
        @type device: string
        """
        self.id = agent_id
        self.q_function = DQN((action_size+obs_size)*num_agents,device)
        self.q_function.apply(init_weights)

        self.q_function_target = deepcopy(self.q_function)
        self.q_grad = torch.optim.Adam(self.q_function.model.parameters(),lr)

        self.policy = DPN(obs_size,action_size,device)
        self.policy.apply(init_weights)
        self.policy_target = deepcopy(self.policy)
        self.policy_grad = torch.optim.Adam(self.policy.model.parameters(),lr)


        self.loss_q = None
        self.exp_ret = None
        self.device = device

    def get_next_action(self,obs:torch.tensor ,explore:bool = False):
        """
        Predict next action using agent policy network
        @param obs: tensor containing observation of the current state of environment
        @type obs: n-dim tensor
        @param explore: whether agent is exploring
        @type explore: bool
        @return: tensor containing probabilities of actions
        @rtype: m-dim tensor
        """
        if explore:
            act = self.policy.forward(obs)
            add_rand = torch.tensor(numpy.random.normal(0,1,size=(len(act)))).to(self.device)
            act = act + add_rand
            act = torch.tensor(act).to(self.device,dtype=torch.float32)
            act = torch.sigmoid(act)
            return act
        else:
            return  self.policy.forward(obs)

    def get_reward(self,obs:torch.tensor):
        """
        Get the expected reward for the current state of the environment
        @param obs: tensor containing complete state of the environment and actions of all the agents
        @type obs: n-dim tensor
        @return: expected reward
        @rtype: float
        """
        return self.q_function.forward(obs)

    def get_reward_target(self,obs:torch.tensor):
        """
        Return expected rewards using target q-function
        @param obs: tensor containing complete state of the environment and actions of all the agents
        @type obs: n-dim tensor
        @return: expected reward
        @rtype: float
        """
        return self.q_function_target(obs)

    def get_action_target(self,obs:torch.tensor):
        """
        Predicts next action for the agent
        @param obs: tensor containing observations of the agent
        @type obs: n-dim tensor
        @return: probabilities of the actions
        @rtype: m-dim tensor
        """
        return self.policy_target(obs)

    def load_checkpoint(self):
        try:
            self.policy.model.load_state_dict(torch.load(f"./model_parameters/agent{self.id}_policy.pth"))
            self.policy_target.model.load_state_dict(torch.load(f"./model_parameters/agent{self.id}_policy_target.pth"))
            self.exp_ret = torch.load(f"./model_parameters/agent{self.id}_exp_ret.pth")["exp_ret"]

            self.q_function.model.load_state_dict(torch.load(f"./model_parameters/agent{self.id}_q.pth"))
            self.q_function_target.model.load_state_dict(torch.load(f"./model_parameters/agent{self.id}_q_target.pth"))
            self.loss_q = torch.load(f"./model_parameters/agent{self.id}_loss_q.pth")["loss_q"]
        except FileNotFoundError as f:
            print("Couldn't Load checkpoint!", str(f))

    def save_checkpoint(self,loss_q,exp_ret):
        if self.exp_ret is None:
            torch.save(self.policy.model.state_dict(), f"./model_parameters/agent{self.id}_policy.pth")
            torch.save(self.policy_target.model.state_dict(), f"./model_parameters/agent{self.id}_policy_target.pth")
            torch.save({"exp_ret":exp_ret},f"./model_parameters/agent{self.id}_exp_ret.pth")
            self.exp_ret = exp_ret

        elif True:
            torch.save(self.policy.model.state_dict(), f"./model_parameters/agent{self.id}_policy.pth")
            torch.save(self.policy_target.model.state_dict(), f"./model_parameters/agent{self.id}_policy_target.pth")
            torch.save({"exp_ret": exp_ret}, f"./model_parameters/agent{self.id}_exp_ret.pth")
            self.exp_ret = exp_ret

        if self.loss_q is None:
            torch.save(self.q_function.model.state_dict(), f"./model_parameters/agent{self.id}_q.pth")
            torch.save(self.q_function_target.model.state_dict(), f"./model_parameters/agent{self.id}_q_target.pth")
            torch.save({"loss_q": loss_q}, f"./model_parameters/agent{self.id}_loss_q.pth")
            self.loss_q = loss_q
        elif True:
            torch.save(self.q_function.model.state_dict(), f"./model_parameters/agent{self.id}_q.pth")
            torch.save(self.q_function_target.model.state_dict(), f"./model_parameters/agent{self.id}_q_target.pth")
            torch.save({"loss_q": loss_q}, f"./model_parameters/agent{self.id}_loss_q.pth")
            self.loss_q = loss_q


class ConstrainedAgent:
    """
    Contains methods, data and models that is related to agents (such as critic model etc.)
    """

    def __init__(self, agent_id:int, action_size:int, obs_size:int, num_agents:int,device:str="cpu",lr=0.01):
        """
        @param agent_id: unique identifier for the agent
        @type agent_id: integer
        @param action_size: size of action space
        @type action_size: integer
        @param obs_size: size of observation space
        @type obs_size: integer
        @param num_agents: number of independent agent in the environment
        @type num_agents: integer
        @param device: device memory to use
        @type device: string
        """
        self.id = agent_id
        self.q_function_r = DQN((action_size+obs_size)*num_agents,device)
        self.q_function_target_r = deepcopy(self.q_function_r)
        self.q_grad_r = torch.optim.Adam(self.q_function_r.model.parameters(),lr)

        self.q_function_c = DQN((action_size + obs_size) * num_agents, device)
        self.q_function_target_c = deepcopy(self.q_function_c)
        self.q_grad_c = torch.optim.Adam(self.q_function_c.model.parameters(), lr)

        self.policy = DPN(obs_size,action_size,device)
        self.policy_target = deepcopy(self.policy)
        self.policy_grad = torch.optim.Adam(self.policy.model.parameters())


        self.loss_q_c = None
        self.loss_q_r = None
        self.exp_ret = None
        self.device = device

    def get_next_action(self,obs:torch.tensor ,explore:bool = False):
        """
        Predict next action using agent policy network
        @param obs: tensor containing observation of the current state of environment
        @type obs: n-dim tensor
        @param explore: whether agent is exploring
        @type explore: bool
        @return: tensor containing probabilities of actions
        @rtype: m-dim tensor
        """
        if explore:
            act = self.policy.forward(obs)
            add_rand = torch.tensor(numpy.random.normal(0,1,size=(len(act)))).to(self.device)
            act = act + add_rand
            act = torch.tensor(act).to(self.device,dtype=torch.float32)
            act = torch.sigmoid(act)
            return act
        else:
            return  self.policy.forward(obs)

    def get_reward(self,obs:torch.tensor):
        """
        Get the expected reward for the current state of the environment
        @param obs: tensor containing complete state of the environment and actions of all the agents
        @type obs: n-dim tensor
        @return: expected reward
        @rtype: float
        """
        return self.q_function_r.forward(obs)



    def get_cost(self,obs:torch.tensor):
        """

        @param obs:
        @type obs:
        @return:
        @rtype:
        """
        return self.q_function_c.forward(obs)


    def get_reward_target(self,obs:torch.tensor):
        """
        Return expected rewards using target q-function
        @param obs: tensor containing complete state of the environment and actions of all the agents
        @type obs: n-dim tensor
        @return: expected reward
        @rtype: float
        """
        return self.q_function_target_r(obs)

    def get_cost_target(self,obs:torch.tensor):
        """

        @param obs:
        @type obs:
        @return:
        @rtype:
        """
        return self.q_function_target_c(obs)

    def get_action_target(self,obs:torch.tensor):
        """
        Predicts next action for the agent
        @param obs: tensor containing observations of the agent
        @type obs: n-dim tensor
        @return: probabilities of the actions
        @rtype: m-dim tensor
        """
        return self.policy_target(obs)

    def load_checkpoint(self):
        try:
            self.policy.model.load_state_dict(torch.load(f"./model_parameters/agent{self.id}_policy.pth"))
            self.policy_target.model.load_state_dict(torch.load(f"./model_parameters/agent{self.id}_policy_target.pth"))
            self.exp_ret = torch.load(f"./model_parameters/agent{self.id}_exp_ret.pth")["exp_ret"]

            self.q_function_r.model.load_state_dict(torch.load(f"./model_parameters/agent{self.id}_q_r.pth"))
            self.q_function_target_r.model.load_state_dict(torch.load(f"./model_parameters/agent{self.id}_q_target_r.pth"))
            self.loss_q_r = torch.load(f"./model_parameters/agent{self.id}_loss_q_r.pth")["loss_q_r"]

            self.q_function_c.model.load_state_dict(torch.load(f"./model_parameters/agent{self.id}_q_c.pth"))
            self.q_function_target_c.model.load_state_dict(torch.load(f"./model_parameters/agent{self.id}_q_target_c.pth"))
            self.loss_q_c = torch.load(f"./model_parameters/agent{self.id}_loss_q_c.pth")["loss_q_c"]
        except FileNotFoundError as f:
            print("Couldn't Load checkpoint!", str(f))

    def save_checkpoint(self,loss_q_r,loss_q_c,exp_ret):

        torch.save(self.policy.model.state_dict(), f"./model_parameters/agent{self.id}_policy.pth")
        torch.save(self.policy_target.model.state_dict(), f"./model_parameters/agent{self.id}_policy_target.pth")
        torch.save({"exp_ret":exp_ret},f"./model_parameters/agent{self.id}_exp_ret.pth")
        self.exp_ret = exp_ret

        torch.save(self.q_function_r.model.state_dict(), f"./model_parameters/agent{self.id}_q_r.pth")
        torch.save(self.q_function_target_r.model.state_dict(), f"./model_parameters/agent{self.id}_q_target_r.pth")
        torch.save({"loss_q_r": loss_q_r}, f"./model_parameters/agent{self.id}_loss_q_r.pth")
        self.loss_q_r = loss_q_r

        torch.save(self.q_function_c.model.state_dict(), f"./model_parameters/agent{self.id}_q_c.pth")
        torch.save(self.q_function_target_c.model.state_dict(), f"./model_parameters/agent{self.id}_q_target_c.pth")
        torch.save({"loss_q_c": loss_q_c}, f"./model_parameters/agent{self.id}_loss_q_c.pth")
        self.loss_q_c = loss_q_c