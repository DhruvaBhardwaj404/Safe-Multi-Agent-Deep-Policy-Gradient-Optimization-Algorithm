import torch
from torch import nn
import typing

class DQN(nn.Module):

    def __init__(self, input_dim:int):
        super().__init__()
        h1  = int(input_dim/3)
        self.model = nn.Sequential( nn.Linear(input_dim,h1),
                                    nn.ReLU(),
                                    nn.Linear(h1,1))


    def forward(self, x):
        reward = self.model(x)
        return reward


