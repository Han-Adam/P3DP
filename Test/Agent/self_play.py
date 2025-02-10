import copy
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from .network import DoubleQNet, PolicyNet
from .replayBuffer import ReplayBuffer
import numpy as np
import json


class TestAgent:
    def __init__(self,
                 path,
                 s_dim=13,
                 a_num=8,
                 hidden=128,):
        # Parameter Initialization
        self.path = path
        self.s_dim = s_dim
        self.a_num = a_num
        self.hidden = hidden
        self.Pi = PolicyNet(s_dim, hidden, a_num)

    def get_action(self, s):
        with torch.no_grad():
            s1, s2 = s
            s2 = torch.tensor(s2, dtype=torch.float)

            prob2_weight = self.Pi(s2)
            action2 = (torch.argmax(prob2_weight)).item()
        return [0, action2]

    def load_net(self, prefix1, prefix2):
        self.Pi.load_state_dict(torch.load(self.path + '/' + prefix1 + '_' + prefix2 + '_Pi_Net.pth'))


class TestAgent_SelfPlay:
    def __init__(self,
                 path,
                 s_dim=13,
                 a_num=8,
                 hidden=128,):
        # Parameter Initialization
        self.path = path
        self.s_dim = s_dim
        self.a_num = a_num
        self.hidden = hidden
        self.Pi1 = PolicyNet(s_dim, hidden, a_num)
        self.Pi2 = PolicyNet(s_dim, hidden, a_num)

    def get_action(self, s):
        with torch.no_grad():
            s1, s2 = s
            s1 = torch.tensor(s1, dtype=torch.float)
            s2 = torch.tensor(s2, dtype=torch.float)

            prob1_weight = self.Pi1(s2)
            action1 = (torch.argmax(prob1_weight)).item()

            prob2_weight = self.Pi2(s1)
            action2 = (torch.argmax(prob2_weight)).item()
        return [action1, action2]

    def load_net_1(self, prefix1, prefix2):
        self.Pi1.load_state_dict(torch.load(self.path + '/' + prefix1 + '_' + prefix2 + '_Pi_Net.pth'))

    def load_net_2(self, prefix1, prefix2):
        self.Pi2.load_state_dict(torch.load(self.path + '/' + prefix1 + '_' + prefix2 + '_Pi_Net.pth'))

