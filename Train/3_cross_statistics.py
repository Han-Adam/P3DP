import numpy as np

from Env import Env
from Agent import TestAgent
import os
import json
import time
import matplotlib.pyplot as plt
import copy
from multiprocessing import Process


strategy_number = [[0, [False, False]],
                   [1, [False, True]],
                   [2, [True, False]],
                   [3, [True, True]]]

best_index = [[1, 40, 0,],
                [4, 29, 0],
                [4, 27, 0],
                [6, 36, 0],
                [1, 30, 0],
                [1, 38, 0],
                [4, 20, 0],
                [6, 33, 0],
                [6, 39, 0],
                [1, 31, 0],]


def statistics_cross(index):
    index1, index2, index3 = best_index[index]
    path_base = os.path.dirname(os.path.realpath(__file__))
    path = path_base + '/Record' + str(index1)
    if not os.path.exists(path):
        os.makedirs(path)

    with open('./test_record.json', 'r') as f:
        env_set = json.load(f)

    agent = TestAgent(path)
    agent.load_net(prefix1=str(index2*10), prefix2=str(index3))
    env = Env()
    strategy_number = [[0, [False, False]],
                       [1, [False, True]],
                       [2, [True, False]],
                       [3, [True, True]]]

    record = np.zeros(shape=[4, 100])

    for opponent_index in range(4):
        strategy_num = strategy_number[opponent_index]
        for env_index in range(100):
            print(opponent_index, env_index)
            s = env.reset(initial_condition=env_set[env_index], strategy_num=strategy_num[1], train=False)
            for ep_step in range(300):
                a = agent.get_action(s)
                s_, r, done = env.step(a)
                done_sum = done[0] or done[1]
                if done_sum:
                    break
                s = s_
            if done_sum:
                if env.red_blood <= 0:
                    if env.blue_blood <= 0:
                        record[opponent_index, env_index] = 0  # tight
                    else:
                        record[opponent_index, env_index] = 1  # win
                else:
                    record[opponent_index, env_index] = -1  # loss
            else:
                if env.red_blood < env.blue_blood:
                    record[opponent_index, env_index] = 1  # win
                elif env.red_blood > env.blue_blood:
                    record[opponent_index, env_index] = -1  # loss
                else:
                    record[opponent_index, env_index] = 0  # tight
        print(opponent_index, np.sum(record[opponent_index]))
    np.save(path_base + '/BestPerform/' + str(index1) +
            '_' + str(index2) + '_' + str(index3) + '_statistics_record_cross.npy', record)


def cross_sum():
    path_base = os.path.dirname(os.path.realpath(__file__))
    record = np.zeros(shape=[10, 4, 100])
    for index in range(10):
        index1, index2, index3 = best_index[index]
        record[index, :, :] = np.load(path_base + '/BestPerform/' + str(index1) +
                                      '_' + str(index2) + '_' + str(index3) + '_statistics_record_cross.npy')
    np.save(path_base + '/BestPerform/statistics_record_cross.npy', record)



if __name__ == '__main__':
    # for i in range(10):
    #     p = Process(target=statistics_cross, args=(i,))
    #     p.start()
    cross_sum()