import torch
import torch.nn as nn


def init_linear(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.1)


class QNet(nn.Module):
    def __init__(self, s_dim, hidden, a_num):
        super(QNet, self).__init__()
        self.feature = nn.Sequential(nn.Linear(s_dim, hidden),
                                     nn.ReLU(),
                                     # nn.Linear(hidden, hidden * 2),
                                     # nn.ReLU(),
                                     nn.Linear(hidden, hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, a_num))

    def forward(self, s):
        return self.feature(s)


class DoubleQNet(nn.Module):
    def __init__(self, s_dim, hidden, a_num):
        super(DoubleQNet, self).__init__()
        self.feature1 = nn.Sequential(nn.Linear(s_dim, hidden),
                                      nn.ReLU(),
                                      # nn.Linear(hidden, hidden),
                                      # nn.ReLU(),
                                      nn.Linear(hidden, hidden),
                                      nn.ReLU(),
                                      nn.Linear(hidden, a_num))
        self.feature2 = nn.Sequential(nn.Linear(s_dim, hidden),
                                      nn.ReLU(),
                                      # nn.Linear(hidden, hidden),
                                      # nn.ReLU(),
                                      nn.Linear(hidden, hidden),
                                      nn.ReLU(),
                                      nn.Linear(hidden, a_num))

    def forward(self, s):
        return self.feature1(s), self.feature2(s)


class PolicyNet(nn.Module):
    def __init__(self, s_dim, hidden, a_num):
        super(PolicyNet, self).__init__()
        self.feature = nn.Sequential(nn.Linear(s_dim, hidden),
                                     nn.ReLU(),
                                     # nn.Linear(hidden, hidden),
                                     # nn.ReLU(),
                                     nn.Linear(hidden, hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, a_num),
                                     nn.Softmax(dim=-1))

    def forward(self, s):
        return self.feature(s)
