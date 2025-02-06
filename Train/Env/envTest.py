import numpy as np
from .f16Model import F16
from .controller import Controller
from .strategy import Strategy
from .util import feet2meter, rad2degree, xyz2llh, angle_error


class EnvTest:
    def __init__(self, time_step=0.02):
        self.time_step = time_step
        self.red_fighter = F16(time_step=self.time_step)
        self.blue_fighter = F16(time_step=self.time_step)
        self.red_controller = Controller(time_step=self.time_step)
        self.blue_controller = Controller(time_step=self.time_step)
        self.red_strategy = Strategy()
        self.blue_strategy = Strategy()

        self.red_blood = self.red_blood_last = 1
        self.blue_blood = self.blue_blood_last = 1
        # only use for testing
        self.record = []
        self.record2 = []

    def reset(self, initial_condition=None, strategy_num=None):
        if initial_condition is None:
            self.red_fighter.reset(position=np.array([np.random.rand() * 6000 - 3000,
                                                      np.random.rand() * 6000 - 3000,
                                                      np.random.rand() * 5000 - 8000]),
                                   euler=np.array([0, -0, (np.random.rand() * 2 - 1) * 180]),
                                   mach=0.9 - np.random.rand() * 0.6)
            self.blue_fighter.reset(position=np.array([np.random.rand() * 6000 - 3000,
                                                       np.random.rand() * 6000 - 3000,
                                                       np.random.rand() * 5000 - 8000]),
                                    euler=np.array([0, 0, (np.random.rand() * 2 - 1) * 180]),
                                    mach=0.9 - np.random.rand() * 0.6)
        else:
            self.red_fighter.reset(position=np.array(initial_condition[0][0]),
                                   euler=np.array(initial_condition[0][1]),
                                   mach=initial_condition[0][2])
            self.blue_fighter.reset(position=np.array(initial_condition[1][0]),
                                    euler=np.array(initial_condition[1][1]),
                                    mach=initial_condition[1][2])
        red_position = self.red_fighter.position * feet2meter
        blue_position = self.blue_fighter.position * feet2meter
        los = red_position - blue_position
        distance = (los[0] ** 2 + los[1] ** 2 + los[2] ** 2) ** 0.5
        if distance < 30:
            return None
        red_ata = np.arccos(np.dot(self.red_fighter.heading, -los) / distance) * rad2degree
        blue_ata = np.arccos(np.dot(self.blue_fighter.heading, los) / distance) * rad2degree
        self.record = []
        self.record2 = []
        self.record.append([[self.red_fighter.position, self.red_fighter.euler, self.red_fighter.mach, 0],
                            [self.blue_fighter.position, self.blue_fighter.euler, self.blue_fighter.mach, 0]])
        self.record2.append([[(self.red_fighter.position * feet2meter).tolist(),
                              (self.red_fighter.euler * rad2degree).tolist(),
                              self.red_fighter.mach, self.red_fighter.height, red_ata, 0],
                             [(self.blue_fighter.position * feet2meter).tolist(),
                              (self.blue_fighter.euler * rad2degree).tolist(),
                              self.blue_fighter.mach, self.blue_fighter.height, blue_ata, 0],
                             distance])
        self.red_controller.reset()
        self.blue_controller.reset()
        if strategy_num is None:
            self.red_strategy.reset(strategy_num=[False, False])
            self.blue_strategy.reset(strategy_num=[False, False])
        else:
            self.red_strategy.reset(strategy_num=strategy_num)
            self.blue_strategy.reset(strategy_num=strategy_num)
        self.red_blood = self.red_blood_last = 1
        self.blue_blood = self.blue_blood_last = 1
        state, _ = self._state_reward(red_done=False, blue_done=False)
        return state

    def step(self, action):
        # red_mode = action[0] + 1 #
        red_mode = self.red_strategy.process(self_fighter=self.red_fighter, target_fighter=self.blue_fighter)
        blue_mode = action[1] + 1
        red_position = self.red_fighter.position * feet2meter
        blue_position = self.blue_fighter.position * feet2meter

        red_done = blue_done = False
        for i in range(50):
            red_u = self.red_controller.control(f16=self.red_fighter, position_target=blue_position, mode=red_mode)
            blue_u = self.blue_controller.control(f16=self.blue_fighter, position_target=red_position, mode=blue_mode)
            self.red_fighter.step(u=red_u)
            self.blue_fighter.step(u=blue_u)
            self.record.append([[self.red_fighter.position, self.red_fighter.euler, self.red_fighter.mach, red_mode],
                                [self.blue_fighter.position, self.blue_fighter.euler, self.blue_fighter.mach,
                                 blue_mode]])

            # termination
            red_position = self.red_fighter.position * feet2meter
            blue_position = self.blue_fighter.position * feet2meter
            los = red_position - blue_position
            distance = (los[0] ** 2 + los[1] ** 2 + los[2] ** 2) ** 0.5
            red_ata = np.arccos(np.dot(self.red_fighter.heading, -los) / distance) * rad2degree
            blue_ata = np.arccos(np.dot(self.blue_fighter.heading, los) / distance) * rad2degree
            self.record2.append([[(self.red_fighter.position * feet2meter).tolist(),
                                  (self.red_fighter.euler * rad2degree).tolist(),
                                  self.red_fighter.mach, self.red_fighter.height, red_ata, red_mode],
                                 [(self.blue_fighter.position * feet2meter).tolist(),
                                  (self.blue_fighter.euler * rad2degree).tolist(),
                                  self.blue_fighter.mach, self.blue_fighter.height, blue_ata, blue_mode],
                                 distance])
            # crash
            if distance < 30:
                self.red_blood = self.blue_blood = 0
                red_done = blue_done = True
                break
            # stall or drop
            red_done = self.red_fighter.alpha * rad2degree > 45 or self.red_fighter.height * feet2meter < 30
            blue_done = self.blue_fighter.alpha * rad2degree > 45 or self.blue_fighter.height * feet2meter < 30
            if red_done or blue_done:
                self.red_blood = 0 if red_done else self.red_blood
                self.blue_blood = 0 if blue_done else self.blue_blood
                break
            # shooting
            if 100 < distance < 1000:
                red_ata = np.arccos(np.dot(self.red_fighter.heading, -los) / distance) * rad2degree
                blue_ata = np.arccos(np.dot(self.blue_fighter.heading, los) / distance) * rad2degree
                if red_ata < 2:
                    self.blue_blood -= self.time_step  # (1000 - distance) / 900 * self.time_step
                if blue_ata < 2:
                    self.red_blood -= self.time_step  # (1000 - distance) / 900 * self.time_step
                red_done = self.red_blood <= 0
                blue_done = self.blue_blood <= 0
                if red_done or blue_done:
                    break
        state, reward = self._state_reward(red_done=red_done, blue_done=blue_done)
        return state, reward, [red_done, blue_done]

    def _state_reward(self, red_done, blue_done):
        """state construction"""
        # calculation from perspective of blue player
        # line of sight, los
        los = (self.red_fighter.position - self.blue_fighter.position) * feet2meter
        distance = (los[0] ** 2 + los[1] ** 2 + los[2] ** 2) ** 0.5
        # heading crossing angle, hca
        hca = np.arccos(np.dot(self.red_fighter.heading, self.blue_fighter.heading))
        # antenna train angle, ata
        ata = np.arccos(np.dot(self.blue_fighter.heading, los) / distance)
        # aspect angle, aa
        aa = np.arccos(np.dot(self.red_fighter.heading, los) / distance)
        # desired pitch
        pitch_d = np.arctan2(- los[2], (los[0] ** 2 + los[1] ** 2) ** 0.5)
        # desired yaw
        yaw_d = np.arctan2(los[1], los[0]) * rad2degree

        angle_self_blue = np.array([self.blue_fighter.euler[0],
                                    self.blue_fighter.euler[1],
                                    self.blue_fighter.path_pitch])
        angle_self_red = np.array([self.red_fighter.euler[0],
                                   self.red_fighter.euler[1],
                                   self.red_fighter.path_pitch])
        angle_relation1_blue = np.array([angle_error(angle=self.blue_fighter.euler[2] * rad2degree, angle_des=yaw_d),
                                         angle_error(angle=self.blue_fighter.path_yaw * rad2degree, angle_des=yaw_d)])
        angle_relation1_red = np.array([angle_error(angle=self.red_fighter.euler[2] * rad2degree, angle_des=yaw_d+180),
                                        angle_error(angle=self.red_fighter.path_yaw * rad2degree, angle_des=yaw_d+180)])
        angle_relation2_blue = np.array([pitch_d, hca, ata, aa])
        angle_relation2_red = np.array([-pitch_d, hca, -aa, -ata])
        # state length = 2 + 3 + 2 + 4 + 2 = 13
        state_blue = np.concatenate([np.array([self.blue_fighter.height * feet2meter / 2000,
                                               self.blue_fighter.ground_speed * feet2meter / 200]),
                                               angle_self_blue / np.pi,
                                               angle_relation1_blue / 180,
                                               angle_relation2_blue / np.pi,
                                               np.array([distance / 2000,
                                               self.red_fighter.ground_speed * feet2meter / 200])])
        state_red = np.concatenate([np.array([self.red_fighter.height * feet2meter / 2000,
                                              self.red_fighter.ground_speed * feet2meter / 200]),
                                              angle_self_red / np.pi,
                                              angle_relation1_red / 180,
                                              angle_relation2_red / np.pi,
                                              np.array([distance / 2000,
                                              self.blue_fighter.ground_speed * feet2meter / 200])])

        """reward calculation"""
        reward = 0
        if red_done or blue_done:
            if red_done:
                reward += 20
            if blue_done:
                reward -= 20
        else:
            reward += (self.red_blood_last - self.red_blood)
            reward -= (self.blue_blood_last - self.blue_blood)
            reward += (np.pi - ata - aa) / np.pi
        reward_blue = reward
        reward_red = 2 - reward_blue  # angle based reward bring bias
        self.red_blood_last = self.red_blood
        self.blue_blood_last = self.blue_blood
        return [state_red, state_blue], [reward_red, reward_blue]

    def save_record(self):
        with open('./record.txt', 'w') as f:
            f.writelines(['FileType=text/acmi/tacview\n',
                          'FileVersion=2.1\n',
                          '0,ReferenceTime=2022-10-01T00:00:00Z\n',
                          '0,Title = test simple aircraft\n'])
            for i in range(len(self.record)):
                f.write('#'+str(self.time_step*(i+1))+'\n')
                red_position = xyz2llh(self.record[i][0][0] * feet2meter)
                red_euler = self.record[i][0][1] * rad2degree
                f.writelines(['1,T=',
                              str(red_position[0]) + '|',
                              str(red_position[1]) + '|',
                              str(red_position[2]) + '|',
                              str(red_euler[0]) + '|',
                              str(red_euler[1]) + '|',
                              str(red_euler[2]) + ', ',
                              'Type=Air+FixedWing,Coalition=Enemies,Color=Red,Name=F-16,Mach=',
                              str(self.record[i][0][2]) + '\n'])
                blue_position = xyz2llh(self.record[i][1][0] * feet2meter)
                blue_euler = self.record[i][1][1] * rad2degree
                f.writelines(['2,T=',
                              str(blue_position[0]) + '|',
                              str(blue_position[1]) + '|',
                              str(blue_position[2]) + '|',
                              str(blue_euler[0]) + '|',
                              str(blue_euler[1]) + '|',
                              str(blue_euler[2]) + ', ',
                              'Type=Air+FixedWing,Coalition=Allies,Color=Blue,Name=F-16,Mach=',
                              str(self.record[i][1][2]) + '\n'])
            f.close()
