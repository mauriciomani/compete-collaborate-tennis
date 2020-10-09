import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    """Xavier weight inizialization: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    Initizaling weights from a uniform distribution and scaling them"""
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Network(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=200, fc2_units=150, actor=False):
        super(Network, self).__init__()

        self.seed = torch.manual_seed(12)
        self.actor = actor
        if self.actor:
            self.fc1 = nn.Linear(state_size, fc1_units)
        else:
            self.fc1 = nn.Linear((state_size+action_size) * 2, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        if self.actor:
            self.fc3 = nn.Linear(fc2_units, action_size)
        else:
            self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action=None):
        if self.actor:
            # return a vector of the force
            h1 = F.relu(self.fc1(state))

            h2 = F.relu(self.fc2(h1))
            h3 = F.tanh(self.fc3(h2))
            #norm = torch.norm(h3)
            
            # h3 is a 2D vector (a force that is applied to the agent)
            # we bound the norm of the vector to be between 0 and 10
            #return 10.0*(f.tanh(norm))*h3/norm if norm > 0 else 10*h3
            return(h3)
        
        else:
            xs = torch.cat((state, action), dim=1)
            # critic network simply outputs a number
            h1 = F.relu(self.fc1(xs))
            h2 = F.relu(self.fc2(h1))
            h3 = (self.fc3(h2))
            return h3