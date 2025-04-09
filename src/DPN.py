import torch.nn
from torch import nn

class DPN(nn.Module):
    def __init__(self,obs_size,action_size,device):
        super().__init__()
        h1 = int(obs_size/2)
        self.model = torch.nn.Sequential(nn.Linear(obs_size, h1),
                                         nn.ReLU(),
                                         nn.Linear(h1,action_size),
                                         nn.Softmax()
                                         ).to(device)

    def forward(self, X):
        actions = self.model(X)
        return actions
    