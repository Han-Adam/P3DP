import copy
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from .network import DoubleQNet, PolicyNet
from .replayBuffer import ReplayBuffer
import numpy as np
import json


class NetworkSet:
    def __init__(self, s_dim, a_num, hidden, lr, initial_alpha):
        # Network
        self.Q = DoubleQNet(s_dim, hidden, a_num)
        self.Q_target = DoubleQNet(s_dim, hidden, a_num)
        self.opt_Q = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.Q_target.load_state_dict(self.Q.state_dict())

        self.Pi = PolicyNet(s_dim, hidden, a_num)
        self.opt_Pi = torch.optim.Adam(self.Pi.parameters(), lr=lr)

        self.alpha = torch.tensor(initial_alpha, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.alpha], lr=lr)


class SelfPlay:
    def __init__(self,
                 path,
                 s_dim=13,
                 a_num=8,
                 hidden=128,
                 gamma=0.9,
                 capacity=int(1e5),
                 batch_size=128,
                 start_learn=512,
                 lr=1e-4,
                 tau=0.05,
                 target_entropy=0.8 * np.log(8),
                 population_size=3):

        # Parameter Initialization
        self.path = path
        self.s_dim = s_dim
        self.a_num = a_num
        self.hidden = hidden
        self.gamma = gamma
        self.capacity = capacity
        self.batch_size = batch_size
        self.start_learn = start_learn
        self.lr = lr
        self.tau = tau
        self.entropy_bar = 1e-5
        self.train_it = 0
        self.period = 4
        self.target_entropy = target_entropy
        self.target_population_entropy = (self.target_entropy + np.log(a_num)) / 2
        self.population_size = population_size

        # Network
        self.network_set = []
        for i in range(self.population_size):
            self.network_set.append(NetworkSet(s_dim, a_num, hidden, lr, 0.))
        # self.log_beta = torch.tensor(np.log(0.6), requires_grad=True)
        self.beta = torch.tensor(0., requires_grad=True) # self.log_beta.exp().detach().item()
        self.opt_beta = torch.optim.Adam([self.beta], lr=lr)

        # replay buffer, or memory
        self.memory_rl = ReplayBuffer(s_dim, capacity, batch_size)

        self.self_pi_set = [i for i in range(self.population_size)]
        self.existing_Pi_num = 0
        self.Pi_num_set = []       # shape = [existing_pi_num - 1]
        self.Pi_total_count = []   # shape = [population_size, population_size, existing_pi_num - 1]
        self.Pi_win_count = []     # shape = [population_size, population_size, existing_pi_num - 1]
        for i in range(self.population_size):
            self.Pi_total_count.append([])
            self.Pi_win_count.append([])
            for j in range(self.population_size):
                self.Pi_total_count[i].append([])
                self.Pi_win_count[i].append([])

        self.self_Pi = self.network_set[0].Pi
        self.self_index = 0
        self.opponent_Pi = copy.deepcopy(self.network_set[0].Pi)
        self.opponent_index1 = 0
        self.opponent_index2 = 0

    def reset(self, red_blood, blue_blood):
        self.Pi_total_count[self.self_index][self.opponent_index1][self.opponent_index2] += 1
        if blue_blood > red_blood:
            self.Pi_win_count[self.self_index][self.opponent_index1][self.opponent_index2] += 1
        elif blue_blood == red_blood:
            self.Pi_win_count[self.self_index][self.opponent_index1][self.opponent_index2] += 0.5

        # selecting self player
        total_count = np.sum(np.sum(np.array(self.Pi_total_count), axis=-1), axis=-1)
        win_count = np.sum(np.sum(np.array(self.Pi_win_count), axis=-1), axis=-1)
        win_rate = win_count / total_count    # shape = [population_size]
        # print(total_count, win_rate)
        priority = win_rate / np.sum(win_rate)
        self.self_index = np.random.choice(a=self.self_pi_set, p=priority)
        self.self_Pi = self.network_set[self.self_index].Pi

        # selecting opponent player
        win_rate = np.array(self.Pi_win_count[self.self_index]) / np.array(self.Pi_total_count[self.self_index])
        priority = 1 - win_rate    # shape = [population_size, existing_pi_num - 1]
        priority1 = np.sum(priority, axis=-1)     # shape = [population_size]
        priority1 = priority1 / np.sum(priority1)
        self.opponent_index1 = np.random.choice(a=self.self_pi_set, p=priority1)
        priority2 = priority[self.opponent_index1, :]     # shape = [existing_pi_num - 1]
        priority2 = priority2 / np.sum(priority2)
        self.opponent_index2 = np.random.choice(a=self.Pi_num_set, p=priority2)
        self.opponent_Pi.load_state_dict(torch.load(
            self.path + '/'+str(self.opponent_index2 * self.period) + '_' + str(self.opponent_index1) + '_Pi_Net.pth'))

    def get_action(self, s):
        s1, s2 = s
        with torch.no_grad():
            s1 = torch.tensor(s1, dtype=torch.float)
            s2 = torch.tensor(s2, dtype=torch.float)

            prob1_weight = self.opponent_Pi(s1)
            dist1 = Categorical(prob1_weight)
            action1 = (dist1.sample()).detach().item()

            prob2_weight = self.self_Pi(s2)
            dist2 = Categorical(prob2_weight)
            action2 = (dist2.sample()).detach().item()
        return [action1, action2]

    def store_transition(self, s, a, s_, r, done):
        s1, s2 = s
        a1, a2 = a
        s_1, s_2 = s_
        r1, r2 = r
        self.memory_rl.store_transition(s2, a2, s_2, r2, done)

    def learn(self):
        self.train_it += 1
        if self.train_it % (1000 * self.period) == 1:
            self.Pi_num_set.append(self.existing_Pi_num)
            self.existing_Pi_num += 1
            for i in range(self.population_size):
                for j in range(self.population_size):
                    self.Pi_total_count[i][j].append(2)
                    self.Pi_win_count[i][j].append(1)

        entropy_set = []
        alpha_beta_set = []

        index = torch.tensor(range(self.batch_size), dtype=torch.long)
        s, a, s_, r, done = self.memory_rl.get_sample()

        # Q Train
        with torch.no_grad():
            prob_total_ = torch.zeros(size=[self.batch_size, self.a_num])
            for i in range(self.population_size):
                prob_total_ += self.network_set[i].Pi(s_)
            prob_total_ /= self.population_size
            population_entropy_ = -torch.sum(prob_total_ * torch.log(prob_total_ + self.entropy_bar), dim=-1)

        for i in range(self.population_size):
            q1, q2 = self.network_set[i].Q(s)
            q1 = q1[index, a]
            q2 = q2[index, a]
            with torch.no_grad():
                q1_target, q2_target = self.network_set[i].Q_target(s_)
                q_target = torch.min(q1_target, q2_target)
                prob = self.network_set[i].Pi(s_)
                v_ = torch.sum(prob * q_target, dim=-1)
                identity_entropy = -torch.sum(prob * torch.log(prob + self.entropy_bar), dim=-1)
                v_target = v_ + self.network_set[i].alpha * identity_entropy
                if i != 0:
                    v_target += self.beta.detach() * population_entropy_
                td_target = r + (1 - done) * self.gamma * v_target
            loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
            self.network_set[i].opt_Q.zero_grad()
            loss.backward()
            self.network_set[i].opt_Q.step()
            self.soft_update(self.network_set[i].Q_target, self.network_set[i].Q)

        # Policy Train
        loss_record = []
        for i in range(self.population_size):
            prob = self.network_set[i].Pi(s)
            q1, q2 = self.network_set[i].Q(s)
            q = torch.min(q1, q2)
            v_ = torch.sum(prob * q, dim=-1)
            identity_entropy = -torch.sum(prob * torch.log(prob + self.entropy_bar), dim=-1)
            loss = -torch.mean(v_ + self.network_set[i].alpha.detach() * identity_entropy)
            loss_record.append(loss)

        with torch.no_grad():
            prob_total = self.network_set[0].Pi(s)
        for i in range(1, self.population_size):
            prob_total += self.network_set[i].Pi(s)
        prob_total /= self.population_size
        population_entropy = -torch.mean(torch.sum(prob_total * torch.log(prob_total + self.entropy_bar), dim=-1))
        loss_record.append(-self.beta.detach() * population_entropy)

        actor_total_loss = sum(loss_record)
        [self.network_set[i].opt_Pi.zero_grad() for i in range(self.population_size)]
        actor_total_loss.backward()
        for i in range(self.population_size):
            self.network_set[i].opt_Pi.step()

        # alpha update
        for i in range(self.population_size):
            with torch.no_grad():
                prob = self.network_set[i].Pi(s)
                identity_entropy = - torch.mean(torch.sum(prob * torch.log(prob + self.entropy_bar), dim=-1))
            loss = self.network_set[i].alpha * (identity_entropy - self.target_entropy)
            self.network_set[i].opt_alpha.zero_grad()
            loss.backward()
            self.network_set[i].opt_alpha.step()
            # self.network_set[i].alpha = self.network_set[i].log_alpha.exp().detach().item()

            entropy_set.append(identity_entropy.detach().item())
            alpha_beta_set.append(self.network_set[i].alpha.detach().item())

        # beta update
        with torch.no_grad():
            prob_total = torch.zeros(size=[self.batch_size, self.a_num])
            for i in range(self.population_size):
                prob_total += self.network_set[i].Pi(s)
            prob_total /= self.population_size
            population_entropy = -torch.mean(torch.sum(prob_total * torch.log(prob_total + self.entropy_bar), dim=-1))
        loss = self.beta * (population_entropy - self.target_population_entropy)
        self.opt_beta.zero_grad()
        loss.backward()
        self.opt_beta.step()

        entropy_set.append(population_entropy.detach().item())
        alpha_beta_set.append(self.beta.detach().item())

        return entropy_set, alpha_beta_set

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1-self.tau) + param.data * self.tau
            )

    def store_net(self, prefix):
        for i in range(self.population_size):
            torch.save(self.network_set[i].Pi.state_dict(), self.path + '/' + prefix + '_' + str(i) + '_Pi_Net.pth')


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

