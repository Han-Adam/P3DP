import matplotlib.pyplot as plt
from Env import Env
from Agent import SelfPlay, TestAgent
import numpy as np
import os
import time
import json

from multiprocessing import Process


def main(index):
    path = os.path.dirname(os.path.realpath(__file__))
    path = path+'/Record' + str(index)
    if not os.path.exists(path):
        os.makedirs(path)

    agent = SelfPlay(path)
    env = Env()
    episode = 0
    store_index = 0
    win_count = 0
    loss_count = 0
    tight_count = 0
    # warm up iteration, collecting samples
    total_steps = 0
    while total_steps <= 512:
        print(total_steps)
        s = env.reset()
        if s is None:
            continue
        for ep_step in range(300):
            a = agent.get_action(s)
            s_, r, done = env.step(a)
            agent.store_transition(s, a, s_, r, done[0] or done[1])
            s = s_
            total_steps += 1
            if done[0] or done[1]:
                break
    # begin training
    total_steps = 0
    win_loss_tight_record = []
    entropy_record = []
    alpha_beta_record = []
    while agent.train_it <= 401000:
        episode += 1
        attitude_record = 0
        reward_record = []
        s = env.reset()
        if s is None:
            continue
        start_time = time.time()
        for ep_step in range(300):
            a = agent.get_action(s)
            s_, r, done = env.step(a)
            done_sum = done[0] or done[1]
            agent.store_transition(s, a, s_, r, done_sum)
            entropy_set, alpha_beta_set = agent.learn()
            reward_record.append(r)
            s = s_
            total_steps += 1
            if agent.train_it % 1000 == 1:
                win_loss_tight_record.append([win_count, loss_count, tight_count])
                entropy_record.append(entropy_set)
                alpha_beta_record.append(alpha_beta_set)
                agent.store_net(str(store_index))
                store_index += 1
            # wining from the perspective of blue [1] UCAV
            if done_sum:
                if done[0]:
                    if done[1]:
                        tight_count += 1
                    else:
                        win_count += 1
                elif done[1]:
                    loss_count += 1
                print()
                # print(reward_record)
                print(' episode:', episode,
                      'ep_step:', ep_step,
                      ' train_step:', agent.train_it,
                      'left_blood:', env.red_blood, env.blue_blood,
                      ' attitude record: ', attitude_record,
                      ' win,loss,tight: ', win_count, loss_count, tight_count,
                      ' entropy: ', entropy_set, alpha_beta_set, agent.target_entropy, agent.target_population_entropy,
                      time.time() - start_time)
                break
        agent.reset(env.red_blood, env.blue_blood)
    np.save(path + '/win_loss_tight.npy', np.array(win_loss_tight_record))
    np.save(path + '/alpha_beta.npy', np.array(alpha_beta_record))
    np.save(path + '/entropy.npy', np.array(entropy_record))


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
    record = np.zeros(shape=[500, 100])

    reward_record = np.zeros(shape=[500, 100])
    reward_number_record = np.zeros(shape=[500, 100])

    for net_index in range(0, 401, 10):
        agent.load_net(prefix1=str(net_index), prefix2=index2)
        for env_index in range(40):
            strategy_num = int(env_index / 10)
            strategy_num = strategy_number[strategy_num]
            s = env.reset(initial_condition=env_set[env_index], strategy_num=strategy_num[1], train=False)
            for ep_step in range(300):
                a = agent.get_action(s)
                s_, r, done = env.step(a)
                done_sum = done[0] or done[1]

                reward_record[net_index, env_index] += r[1]
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
            reward_number_record[net_index, env_index] = ep_step
        print(index1, index2, net_index, np.sum(record[net_index]))
        np.save(path + '/statistics_record_' + index2 + '.npy', record)
        np.save(path + '/reward_record_' + index2 + '.npy', reward_record)
        np.save(path + '/reward_number_record_' + index2 + '.npy', reward_number_record)



def sequence_statistics(index):
    for i in range(3):
        statistics_single(index, i)


def one_processing(index):
    # main(index)
    # main(index + 4)
    statistics_single(index, 0)
    # statistics_single(0+index*4, 0)
    # statistics_single(1 + index * 4, 0)
    # statistics_single(2 + index * 4, 0)
    # statistics_single(3 + index * 4, 0)
    # # statistics_single(index, 0)
    # # statistics_single(index, 2)
    # # statistics_single(index+2, 1)
    # # statistics_single(index+2, 2)
    # # statistics_single(index + 4, 1)
    # # statistics_single(index + 4, 2)
    # # statistics_single(index + 6, 1)
    # # statistics_single(index + 6, 2)



if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    for i in range(8):
        p = Process(target=one_processing, args=(i,))
        p.start()


    # process_list = []
    # for i in range(12):
    #     p = Process(target=one_processing, args=(i, ))
    #     p.start()
    #     process_list.append(p)
    # one_processing(0)
    # train1
    # process_list = []
    # for i in range(4):
    #     for j in range(3):
    #         p = Process(target=statistics_single, args=(i+4, j))
    #         p.start()
    #         process_list.append(p)

    #
    # for i in range(len(process_list)):
    #     process_list[i].join()

    # process_list = []
    # for i in range(8):
    #     p = Process(target=statistics_single, args=(i, 0))
    #     p.start()
    #     process_list.append(p)
    # for i in range(4):
    #     p = Process(target=statistics_single, args=(i, 1))
    #     p.start()
    #     process_list.append(p)
    #
    # for i in range(len(process_list)):
    #     process_list[i].join()
    #
    # process_list = []
    # for i in range(4):
    #     p = Process(target=statistics_single, args=(i + 4, 1))
    #     p.start()
    #     process_list.append(p)
    # for i in range(8):
    #     p = Process(target=statistics_single, args=(i, 2))
    #     p.start()
    #     process_list.append(p)