from Env import EnvTest
from Agent import TestAgent_SelfPlay
import os
import time
import numpy as np

feet2meter = 0.3048


def main(index, strategy_num):
    path = os.path.dirname(os.path.realpath(__file__))
    path = path + '/Record0'
    if not os.path.exists(path):
        os.makedirs(path)

    agent = TestAgent_SelfPlay(path)
    agent.load_net_1(prefix1=str(9), prefix2=str(0))
    agent.load_net_2(prefix1=str(9), prefix2=str(0))
    env = EnvTest()
    total_steps = 0

    attitude_record = 0
    reward_record = []

    # # case 1
    index = 9
    agent.load_net_1(prefix1=str(index), prefix2=str(0))
    agent.load_net_2(prefix1=str(index), prefix2=str(0))
    initial_condition = [[[0., -1000., -4000.], [0., 0., 0.], 0.9],
                         [[0., 1000., -4000.], [0., 0., 180.], 0.9]]
    strategy_num = [False, False]
    #
    # # case 2
    # initial_condition = [[[0., -1000., -4000.], [0., 0., 0.], 0.9],
    #                      [[0., 1000., -4000.], [0., 0., 180.], 0.9]]
    # strategy_num = [True, True]

    # case 3
    # initial_condition = [[[1000., 0., -4000.], [0., 0., 180.], 0.9],
    #                      [[-1000., 0., -4000.], [0., 0., 180.], 0.9]]
    # strategy_num = [False, False]

    s = env.reset(initial_condition=initial_condition, strategy_num=strategy_num, train=False)
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
