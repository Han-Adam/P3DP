from Env import EnvTest
from Agent import SelfPlay
import os
import time
import numpy as np

feet2meter = 0.3048


def main(index, strategy_num):
    path = os.path.dirname(os.path.realpath(__file__))
    path = path + '/Record' + '_' + str(strategy_num[0]) + '_' + str(index)
    if not os.path.exists(path):
        os.makedirs(path)

    agent = SelfPlay(path)
    agent.load_net(prefix='50')
    env = EnvTest()
    total_steps = 0

    attitude_record = 0
    reward_record = []

    # # case 1
    initial_condition = [[[0., -2000., -5000.], [0., 0., 0.], 0.9],
                         [[0., 2000., -5000.], [0., 0., 180.], 0.9]]
    strategy_num = [True, True]
    # [0, 963.3253733946163, -6497.980116223842], [0, 0, 0.0], 0.695122047800898
    initial_condition = [[[0, 963.3253733946163, -6497.980116223842], [0., 0., 0.], 0.695122047800898],
                         [[0, -963.3253733946163, -6497.980116223842], [0., 0., 180.], 0.695122047800898]]
    strategy_num = [True, False]
    # case 2
    # initial_condition = [[[0., -2000., -5000.], [0., 0., 0.], 0.8],
    #                      [[0., 2000., -5000.], [0., 0., 180.], 0.8]]
    # strategy_num = [True, True, True, True]
    # case 3
    # initial_condition = [[[0., -2000., -5000.], [0., 0., 0.], 0.8],
    #                      [[0., 2000., -5000.], [0., 0., 180.], 0.8]]
    # strategy_num = [False, False, True, True]# [False, False, True, True]
    s = env.reset(initial_condition=initial_condition, strategy_num=strategy_num)
    start_time = time.time()
    for ep_step in range(300):
        a = agent.get_action(s)

        s_, r, done = env.step(a)
        # agent.store_transition(s, a, s_, r, done)
        reward_record.append(r)
        s = s_
        total_steps += 1
        print(ep_step, env.red_blood, env.blue_blood, np.array(a)+1, env.red_fighter.mach, env.blue_fighter.mach)
        if done[0] or done[1]:
            print()
            # print(reward_record)
            print('ep_step:', ep_step,
                  ' train_step:', agent.train_it,
                  'left_blood:', env.red_blood, env.blue_blood,
                  ' attitude record: ', attitude_record,
                  time.time() - start_time)
            break
    env.save_record()


if __name__ == '__main__':
    strategy_number = [[1, [False, False]],
                       [2, [False, True]],
                       [3, [True, False]],
                       [4, [True, True]]]
    # for i in range(1):
    #     main(i, strategy_number[3])
    main(6, strategy_number[3])
