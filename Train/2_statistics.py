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


def statistics_single(index1, index2):
    index1 = str(index1)
    index2 = str(index2)
    path = os.path.dirname(os.path.realpath(__file__))
    path = path+'/Record' + index1
    if not os.path.exists(path):
        os.makedirs(path)

    with open('./test_record.json', 'r') as f:
        env_set = json.load(f)

    agent = TestAgent(path)
    env = Env()
    # env_set = []
    record = np.zeros(shape=[41, 100])

    for net_index in range(0, 41):
    # for net_index in range(0, 401, 10):
        agent.load_net(prefix1=str(int(10 * net_index)), prefix2=index2)
        for env_index in range(100):
            strategy_num = int(env_index / 25)
            strategy_num = strategy_number[strategy_num]
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
                        record[net_index, env_index] = 0  # tight
                    else:
                        record[net_index, env_index] = 1  # win
                else:
                    record[net_index, env_index] = -1  # loss
            else:
                if env.red_blood < env.blue_blood:
                    record[net_index, env_index] = 1  # win
                elif env.red_blood > env.blue_blood:
                    record[net_index, env_index] = -1  # loss
                else:
                    record[net_index, env_index] = 0  # tight
        print(index1, index2, net_index, np.sum(record[net_index]))
        np.save(path + '/statistics_record_' + index2 + '.npy', record)




def sum_record():
    path_base = os.path.dirname(os.path.realpath(__file__))
    # path = path + '/Record' + str(2) + '_' + str(0)

    total_record = np.zeros(shape=[8, 3, 41, 100])

    for index1 in range(8):
        path = path_base + '/Record' + str(index1)
        for index2 in range(3):
            record = np.load(path + '/statistics_record_' + str(index2) + '.npy')
            total_record[index1, index2, :, :] = record[0:401:10, :]

            print(index1, index2, np.sum(total_record[index1, index2, :, :], axis=1))

    np.save(path_base + '/statistics_record_total.npy', total_record)

    reward_record = np.zeros(shape=[8, 3, 41, 100])
    reward_number_record = np.zeros(shape=[8, 3, 41, 100])
    for index1 in range(8):
        path = path_base + '/Record' + str(index1)
        for index2 in range(1):
            record = np.load(path + '/reward_record_' + str(index2) + '.npy')
            reward_record[index1, index2, :, :] = record[0:401:10, :]

            record_number = np.load(path + '/reward_number_record_' + str(index2) + '.npy')
            reward_number_record[index1, index2, :, :] = record_number[0:401:10, :]

    np.save(path_base + '/reward_record_P3DP.npy', reward_record)
    np.save(path_base + '/reward_number_record_P3DP.npy', reward_number_record)


def entropy_record():
    path_base = os.path.dirname(os.path.realpath(__file__))
    # path = path + '/Record' + str(2) + '_' + str(0)

    total_record = np.zeros(shape=[8, 400, 4])

    for index in range(8):
        path = path_base + '/Record' + str(index)
        record = np.load(path + '/entropy.npy')
        total_record[index, :, :] = record[0:400, :]
    np.save(path_base + '/entropy_total.npy', total_record)

    total_record = np.zeros(shape=[8, 400, 4])

    for index in range(8):
        path = path_base + '/Record' + str(index)
        record = np.load(path + '/alpha_beta.npy')
        total_record[index, :, :] = record[0:400, :]
    np.save(path_base + '/alpha_beta_total.npy', total_record)

    total_record = np.zeros(shape=[8, 400, 3])

    for index in range(8):
        path = path_base + '/Record' + str(index)
        record = np.load(path + '/win_loss_tight.npy')
        total_record[index, :, :] = record[0:400, :]
    np.save(path_base + '/win_loss_tight.npy', total_record)


if __name__ == '__main__':
    entropy_record()
    sum_record()
