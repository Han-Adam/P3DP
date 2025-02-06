from scipy.spatial.transform import Rotation as R
import numpy as np
from .fForward import fForward
from .adc import adc
from .util import meter2feet, degree2rad


class F16:
    def __init__(self, time_step=0.01):
        self.time_step = time_step                     # time step for Simulation (s)

        self.position = np.zeros(shape=[3])            # x, y, z, (feet)
        self.euler = np.zeros(shape=[3])               # roll, pitch, yaw; phi, theta, psi, (rad)
        self.velocity = np.zeros(shape=[3])            # u, v, w, velocity under body frame (ft/s)
        self.angular_velocity = np.zeros(shape=[3])    # p, q, r, angular velocity under body frame (rad/s)

        self.height = 0                                # height (feet)
        self.ground_speed = 0                          # ground speed = || velocity || (ft/s)
        self.mach = 0                                  # mach = ground speed / sound speed
        self.alpha = 0                                 # attack angle (rad)
        self.beta = 0                                  # sideslip angle (rad)
        self.path_pitch = 0                            # path_pitch (rad)
        self.path_yaw = 0                              # path_yaw (rad)
        self.elevator = 0                              # elevator, [-25, 25], rate restriction: 60
        self.elevator_restriction = 60 * time_step
        self.aileron = 0                               # aileron, [-21.5, 21.5], rate restriction: 80
        self.aileron_restriction = 80 * time_step
        self.rudder = 0                                # rudder, [-30, 30], rate restriction: 120
        self.rudder_restriction = 120 * time_step
        self.power = 0                                 # engine thrust dynamics lag state, power
        self.overload_y = 0                            # radial overload
        self.overload_z = 0                            # normal overload
        self.qbar = 0                                  # dynamic pressure

        self.rotation_body2earth = np.eye(N=3)
        self.heading = self.rotation_body2earth[:, 0]
        self.velocity_earth = np.zeros(shape=[3])
        self.rotation_trajectory2earth = np.eye(N=3)
        self.rotation_velocity2earth = np.eye(N=3)

    def reset(self, position=np.array([0., 0., 0.]), euler=np.array([0., 0., 0.]), mach=0.5):
        self.height = -position[2] * meter2feet
        temperature = 390 if self.height >= 35000 else 519 * (1 - .703e-5 * self.height)
        sound_speed = (1.4 * 1716.3 * temperature) ** 0.5
        velocity = sound_speed * mach

        self.position = position * meter2feet
        self.euler = euler * degree2rad
        self.velocity = np.array([velocity, 0, 0])
        self.angular_velocity = np.zeros(shape=[3])
        self.ground_speed = velocity
        self.mach, self.qbar = adc(groundSpeed=velocity, height=self.height)
        self._update_rotation()
        self.elevator = 0
        self.aileron = 0
        self.rudder = 0
        self.power = 50

    def step(self, u):
        # update control input
        throttle, elevator_command, aileron_command, rudder_command = u
        throttle = np.clip(throttle, 0, 1)
        # elevator
        elevator_command = np.clip(elevator_command, -25, 25)
        elevator_diff = elevator_command - self.elevator
        if abs(elevator_diff) > self.elevator_restriction:
            self.elevator += np.sign(elevator_diff) * self.elevator_restriction
        else:
            self.elevator = elevator_command
        # aileron
        aileron_command = np.clip(aileron_command, -21.5, 21.5)
        aileron_diff = aileron_command - self.aileron
        if abs(aileron_diff) > self.aileron_restriction:
            self.aileron += np.sign(aileron_diff) * self.aileron_restriction
        else:
            self.aileron = aileron_command
        # rudder
        rudder_command = np.clip(rudder_command, -30, 30)
        rudder_diff = rudder_command - self.rudder
        if abs(rudder_diff) > self.rudder_restriction:
            self.rudder += np.sign(rudder_diff) * self.rudder_restriction
        else:
            self.rudder = rudder_command

        u = [throttle, self.elevator, self.aileron, self.rudder]

        # construct state
        state = np.concatenate([self.position, self.velocity, self.euler, self.angular_velocity, [self.power]])
        # classic 4-order Runge-Kutta
        # k1, overload_y1, overload_z1 = fForward(state=state, u=u)
        # k2, overload_y2, overload_z2 = fForward(state=state + k1 * self.time_step / 2, u=u)
        # k3, overload_y3, overload_z3 = fForward(state=state + k2 * self.time_step / 2, u=u)
        # k4, overload_y4, overload_z4 = fForward(state=state + k3 * self.time_step, u=u)
        # state_ = state + ((k1 + k4) / 6 + (k2 + k3) / 3) * self.time_step
        # self.overload_y = ((overload_y1 + overload_y4) / 6 + (overload_y2 + overload_y3) / 3)
        # self.overload_z = ((overload_z1 + overload_z4) / 6 + (overload_z2 + overload_z3) / 3)
        k, self.overload_y, self.overload_z = fForward(state=state, u=u)
        state_ = state + k * self.time_step

        self.position = state_[0: 3]
        self.velocity = state_[3: 6]
        self.euler = state_[6: 9]
        self.angular_velocity = state_[9: 12]
        self.power = state_[12]
        self.ground_speed = np.linalg.norm(self.velocity, ord=2)
        self.height = -self.position[2]
        self.mach, self.qbar = adc(groundSpeed=self.ground_speed, height=self.height)
        self._update_rotation()

    def _update_rotation(self):
        self.rotation_body2earth = R.from_euler(seq='ZYX', angles=self.euler[::-1]).as_matrix()
        # x = R.from_matrix(self.rotation_body2earth).as_euler(seq='ZYX')
        self.euler = R.from_matrix(self.rotation_body2earth).as_euler(seq='ZYX')[::-1]
        self.heading = self.rotation_body2earth[:, 0]
        self.velocity_earth = np.matmul(self.rotation_body2earth, self.velocity)
        self.path_pitch = np.arctan2(-self.velocity_earth[2],
                                     (self.velocity_earth[0] ** 2 + self.velocity_earth[1] ** 2) ** 0.5)
        self.path_yaw = np.arctan2(self.velocity_earth[1], self.velocity_earth[0])
        self.rotation_trajectory2earth = R.from_euler(seq='ZY', angles=[self.path_yaw, self.path_pitch]).as_matrix()
        self.rotation_velocity2earth = R.from_euler(seq='ZYX', angles=[self.path_yaw, self.path_pitch,
                                                                       self.euler[0]]).as_matrix()
        self.alpha = np.arctan2(self.velocity[2], self.velocity[0])
        self.beta = np.arcsin(self.velocity[1] / self.ground_speed)