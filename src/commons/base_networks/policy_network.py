import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import os
import numpy as np

'''
    This is also called the Actor-Network
'''
class BasePolicyNetwork(nn.Module):
    def __init__(self, alpha=0.0001, state_dim=3, action_dim=1, action_range=1, 
            log_std_min=-20, log_std_max=2, hidden_size=128, init_w=3e-3, 
            name='policy', chkpt_dir='tmp/', method=''):
        super(BasePolicyNetwork, self).__init__()

        self.name = name
        self.checkpoint_dir = chkpt_dir + method
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_' + method)
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_dim)
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.mu.bias.data.uniform_(-init_w, init_w)
        
        self.log_std = nn.Linear(hidden_size, action_dim)
        self.log_std.weight.data.uniform_(-init_w, init_w)
        self.log_std.bias.data.uniform_(-init_w, init_w)

        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_range = action_range

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        pass
    
    def sample_normal(self, state, deterministic=False):
        pass

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

    def sample_action(self):
        action = torch.FloatTensor(self.action_dim).uniform_(-1, 1)
        action = self.action_range * action
        action = self.rescale_action(action)
        return (action).numpy()

    def rescale_action(self, action):
        return (action + 1) /2