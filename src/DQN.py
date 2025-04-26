import torch
from torch import nn
import typing

class DQN(nn.Module):

    def __init__(self, input_dim:int,device):
        super().__init__()
        h1 = int(input_dim * 2)
        h2 = int(input_dim)
        self.model = nn.Sequential( nn.Linear(input_dim,h1),
                                    nn.ReLU(),
                                    nn.Linear(h1, h2),
                                    nn.ReLU(),
                                    nn.Linear(h2,1)).to(device)


    def forward(self, x):
        reward = self.model(x)
        return reward


