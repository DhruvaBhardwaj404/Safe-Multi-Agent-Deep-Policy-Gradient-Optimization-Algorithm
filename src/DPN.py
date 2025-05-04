import torch.nn
from torch import nn

class DPN(nn.Module):
    def __init__(self,obs_size,action_size,device):
        super().__init__()
        h1 = int(obs_size*1.5)
        h2 = int(obs_size*1.5)
        self.model = torch.nn.Sequential(nn.Linear(obs_size, h1),
                                         nn.ReLU(),
                                         nn.Linear(h1, h2),
                                         nn.ReLU(),
                                         nn.Linear(h2,action_size),
                                         nn.LogSoftmax()
                                         ).to(device)

    def forward(self, X):
        actions = self.model(X)
        return actions
    