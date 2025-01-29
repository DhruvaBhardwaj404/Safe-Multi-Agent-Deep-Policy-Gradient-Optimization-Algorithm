from copy import deepcopy
import torch
from agilerl.components.replay_buffer import ReplayBuffer
import numpy
import time

from numpy.core.fromnumeric import argmax
from torchviz import make_dot
from src.DQN import DQN
from src.DPN import DPN
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, ListStorage
import warnings


class MADDPG:

    def __init__(self,obs_size:int,action_size:int,agent_num:int,gamma:float, eps:float,tau:float,threshold:int,device:str, eps_decay:int=1000):
        torch.autograd.set_detect_anomaly(True)
        warnings.filterwarnings("ignore")
        self.action_size = action_size
        self.obs_size = obs_size
        self.agent_num = agent_num
        self.q_function= []
        self.q_function_target = None
        self.policy = []
        self.gamma = torch.tensor(gamma).to(device)
        self.eps = torch.tensor(eps).to(device)
        self.eps_decay = torch.tensor(eps_decay).to(device)
        self.tau = torch.tensor(tau).to(device)
        self.batch_size = 8
        self.replay = ReplayBuffer(storage=ListStorage(max_size=1024),batch_size=1)
        self.q_optim = []
        self.p_optim = []
        self.threshold = threshold
        self.device = device
        self.steps = 0
        self.checkpoint_q_loss = 10e6
        self.checkpoint_p_loss = 10e6
        self.checkpoint_q_time = int(time.time())
        self.checkpoint_p_time = int(time.time())


        for i in range(0,agent_num):
            temp = DQN(obs_size+action_size*self.agent_num).to(self.device)
            self.q_function.append(temp)
            self.q_optim.append(torch.optim.Adam(self.q_function[-1].model.parameters()))
            temp = DPN(obs_size,action_size).to(self.device)
            self.policy.append(temp)
            self.p_optim.append(torch.optim.Adam(self.policy[-1].model.parameters()))


        self.q_function_target = deepcopy(self.q_function)
        self.policy_target = deepcopy(self.policy)
        self.load_checkpoint()

    def one_hot_encode(self, action_prob:torch.tensor):
        """
        Convert a n-dim tensor of probabilities of action to one hot encoded tensor.
        @param action_prob: vector of predicted probabilities from policy of an agent
        @type action_prob: n-dim tensor
        @return: one hot-encoded vector
        @rtype: n-dim tensor
        """

        max_act_ind = torch.argmax(action_prob)
        ohe_vector = [0 for _ in range(0,self.action_size)]
        ohe_vector[max_act_ind] = 1
        return ohe_vector


    def add_to_replay(self,obs_old:torch.tensor, action:torch.tensor, reward:torch.tensor,
                      obs_new:torch.tensor):
        if reward.size()[0]==0:
            return
        self.replay.add(TensorDict({"obs":obs_old, "act":action, "rew":reward, "nobs":obs_new}))


    def save_checkpoint(self,q_loss,j_loss,i):
        try:
            if int(time.time()) + 100 > self.checkpoint_q_time:
                self.checkpoint_q_time = int(time.time())
                if q_loss < self.checkpoint_q_loss:
                    torch.save(self.q_function[i].model.state_dict(),f"agent_dqn_{i}.pth")
                    torch.save(self.q_function_target[i].model.state_dict(), f"agent_dqnt_{i}.pth")
                    torch.save({"cqtime":self.checkpoint_q_time, "qloss":self.checkpoint_q_loss},"qcheck.pth")
            if int(time.time()) + 100 > self.checkpoint_p_time:
                self.checkpoint_p_time = int(time.time())
                if j_loss < self.checkpoint_p_loss:
                    torch.save(self.policy[i].model.state_dict(), f"agent_dpn_{i}.pth")
                    torch.save({"cptime":self.checkpoint_p_time,"ploss":self.checkpoint_q_loss}, "pcheck.pth")
        except Exception as e:
            print(f"Couldn't save checkpoint {str(e)}")

    def load_checkpoint(self):
        try:
            for i in range(0,self.agent_num):
                self.q_function[i].model.load_state_dict(torch.load(f"agent_dqn_{i}.pth"))
                self.q_function_target[i].model.load_state_dict(torch.load(f"agent_dqnt_{i}.pth"))
                self.policy[i].model.load_state_dict(torch.load(f"agent_dpn_{i}.pth"))
                pcheck = torch.load("pcheck.pth")
                self.checkpoint_p_time = pcheck["cptime"]
                self.checkpoint_p_loss = pcheck["ploss"]
                qcheck = torch.load("qcheck.pth")
                self.checkpoint_q_time = qcheck["cqtime"]
                self.checkpoint_q_loss = qcheck["qloss"]
        except Exception as e:
            print(f"Couldn't load checkpoint {str(e)}")

    def get_action(self,obs):
        agent_actions = []
        obs = obs.flatten()
        with torch.no_grad():
            sample = numpy.random.random()
            eps_threshold = self.eps + (self.eps - 0.05 ) * \
                            torch.exp(-1. * self.steps/ self.eps_decay)
            self.steps += 1
            for p in self.policy:
                temp = p.forward(obs).to("cpu").numpy()
                if sample > eps_threshold:
                    a = numpy.add(temp, numpy.random.normal(loc=1,scale=0.5,size=self.action_size))
                    a = numpy.argmax(a)
                    agent_actions.append(a)
                else:
                    agent_actions.append(numpy.argmax(temp))

        return agent_actions


    def update(self):
        if len(self.replay) < self.threshold:
            return

        q_loss = []
        p_loss = []

        samples = self.replay.sample(self.batch_size)
        samples.to(self.device)
        loss_q = torch.tensor([1]).to(self.device,dtype=torch.float32)
        loss_j = torch.tensor([1]).to(self.device,dtype=torch.float32)

        for j in range(self.batch_size):
            sample = samples[j]
            obs = sample["obs"]
            act = sample["act"]
            nobs = sample["nobs"]
            rew = sample["rew"]
            for i in range(self.agent_num):
                nact = torch.tensor([ self.one_hot_encode(p.forward(nobs[i])) for p in self.policy_target])

            qtinp = torch.cat((nobs, nact))
            qinp = torch.cat((obs, act))

            for i in range(self.agent_num):
                y =  rew + self.gamma*self.q_function_target[i].forward(qtinp)
                q = self.q_function[i].forward(qinp)





        print("q loss=>", q_loss)
        print("p loss=>", p_loss)

        for i in range(0, self.agent_num):
            with torch.no_grad():
                for qt, q in zip(self.q_function_target[i].model.parameters(),
                                 self.q_function[i].model.parameters()):
                    new = torch.add(torch.mul(self.tau, q), torch.mul(1 - self.tau, qt))
                    qt.copy_(new)
                for pt, p in zip(self.policy_target[i].model.parameters(),
                                 self.policy[i].model.parameters()):
                    new = torch.add(torch.mul(self.tau, p), torch.mul(1 - self.tau, pt))
                    qt.copy_(new)

            self.save_checkpoint(q_loss[i], p_loss[i], i)
        # print(self.q_function_target[0].model.state_dict())
        print(self.q_function[0].model.state_dict())
        # print(self.policy[0].model.state_dict())
