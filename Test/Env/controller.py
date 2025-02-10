import numpy as np
from .util import angle_error, gravity, feet2meter, rad2degree


gravity = gravity[2] * feet2meter


class Controller:
    def __init__(self, time_step):
        self.time_step = time_step
        self.alpha_last = 0
        self.alpha_integral = 0
        self.beta_last = 0
        self.path_pitch_last = 0

        self.mode_last = 0
        self.SP_stage1 = True
        self.SP_stage2 = True

    def reset(self):
        self.alpha_last = 0
        self.alpha_integral = 0
        self.beta_last = 0
        self.path_pitch_last = 0

        self.mode_last = 0
        self.SP_stage1 = True
        self.SP_stage2 = True

    def control(self, f16, position_target, mode):
        """
        1: straight fly
        2: climb
        3: lopping
        4: split_s
        5: attitude_tracking
        6: position_tracking
        7: high yoyo
        8: low yoyo
        """
        if mode == 1:
            self.mode_last = 1
            return self._straight(f16=f16, pitch_des=0)
        if mode == 2:
            self.mode_last = 2
            return self._straight(f16=f16, pitch_des=20)
        if mode == 3:
            self.mode_last = 7
            return self._lopping(f16=f16)
        if mode == 4:
            return self._split_s(f16=f16)

        self.mode_last = mode
        position_self = f16.position * feet2meter
        position_error = position_target - position_self
        if mode == 5:
            pitch_des = np.arctan2(-position_error[2],
                                   (position_error[1] ** 2 + position_error[0] ** 2) ** 0.5) * rad2degree
            yaw_des = np.arctan2(position_error[1], position_error[0]) * rad2degree
            return self._attitude_tracking(f16=f16, pitch_des=pitch_des, yaw_des=yaw_des)
        elif mode == 6:
            pitch_des = np.arctan2(-position_error[2],
                                   (position_error[1] ** 2 + position_error[0] ** 2) ** 0.5) * rad2degree
            yaw_des = np.arctan2(position_error[1], position_error[0]) * rad2degree
            return self._position_tracking(f16=f16, pitch_des=pitch_des, yaw_des=yaw_des)
        else:
            ground_speed = f16.ground_speed * feet2meter
            pitch_des = np.arctan2(-position_error[2] +
                                   (ground_speed ** 2) / (2 * 9.805) * (0.1 if mode == 7 else -0.1),
                                   (position_error[1] ** 2 + position_error[0] ** 2) ** 0.5) * rad2degree
            yaw_des = np.arctan2(position_error[1], position_error[0]) * rad2degree
            return self._position_tracking(f16=f16, pitch_des=pitch_des, yaw_des=yaw_des)

    def _straight(self, f16, pitch_des):
        ground_speed = f16.ground_speed * feet2meter
        roll = f16.euler[0] * rad2degree
        pitch = f16.euler[1] * rad2degree
        alpha = f16.alpha * rad2degree
        beta = f16.beta * rad2degree

        pitch_error = angle_error(angle=pitch, angle_des=pitch_des)
        load = ground_speed * np.clip(0.02 * pitch_error, -1, 1) + 9.81 * np.cos(f16.path_pitch)
        load = np.clip(load / 9.81, -1, 5)
        alpha_des = np.clip(load * 4, -4, 20)
        elevator = self._alpha_control(alpha=alpha, alpha_des=alpha_des)
        aileron = self._roll_control(roll=roll, roll_des=0)
        rudder = self._beta_control(beta=beta, beta_des=0)
        return np.array([1 if f16.mach < 0.9 else 0, elevator, aileron, rudder])

    def _lopping(self, f16):
        roll = f16.euler[0] * rad2degree
        alpha = f16.alpha * rad2degree
        beta = f16.beta * rad2degree

        elevator = self._alpha_control(alpha=alpha, alpha_des=30)
        aileron = self._roll_control(roll=roll, roll_des=0 if abs(roll) < 90 else 180)
        rudder = self._beta_control(beta=beta, beta_des=0)
        return np.array([1, elevator, aileron, rudder])

    def _split_s(self, f16):
        if self.mode_last != 4:
            self.SP_stage1 = True
            self.SP_stage2 = True
        self.mode_last = 4

        roll = f16.euler[0] * rad2degree
        alpha = f16.alpha * rad2degree
        beta = f16.beta * rad2degree

        if self.SP_stage1:
            elevator = self._alpha_control(alpha=alpha, alpha_des=0)
            aileron = self._roll_control(roll=roll, roll_des=180)
            rudder = self._beta_control(beta=beta, beta_des=0)
            if abs(roll) > 170:
                self.SP_stage1 = False
            return np.array([1, elevator, aileron, rudder])
        elif self.SP_stage2:
            roll_des = 180 if abs(roll) > 160 else 0
            elevator = self._alpha_control(alpha=alpha, alpha_des=30)
            aileron = self._roll_control(roll=roll, roll_des=roll_des)
            rudder = self._beta_control(beta=beta, beta_des=0)
            if f16.euler[1] * rad2degree > 2:
                self.SP_stage2 = False
            return np.array([1, elevator, aileron, rudder])
        else:
            return self._straight(f16=f16, pitch_des=0)

    def _attitude_tracking(self, f16, pitch_des, yaw_des):
        roll, pitch, yaw = f16.euler * rad2degree
        alpha = f16.alpha * rad2degree
        beta = f16.beta * rad2degree
        ground_speed = f16.ground_speed * feet2meter
        trajectory2earth = f16.rotation_trajectory2earth
        velocity2earth = f16.rotation_velocity2earth

        pitch_error = angle_error(angle=pitch, angle_des=pitch_des)
        yaw_error = angle_error(angle=yaw, angle_des=yaw_des)
        pitch_load = ground_speed * np.clip(0.02 * pitch_error, -1, 1) + 9.81 * np.cos(f16.path_pitch)
        yaw_load = ground_speed * np.clip(0.02 * yaw_error, -1, 1) * np.cos(f16.path_pitch)

        load_trajectory = np.array([0, yaw_load, -pitch_load])
        load_velocity = np.matmul(velocity2earth.T, np.matmul(trajectory2earth, load_trajectory))
        load = np.clip(-load_velocity[2] / 9.81, -1, 5)
        alpha_des = np.clip(load * 4, -4, 20)
        roll_des = np.arctan2(load_trajectory[1], -load_trajectory[2]) * rad2degree

        elevator = self._alpha_control(alpha=alpha, alpha_des=alpha_des)
        aileron = self._roll_control(roll=roll, roll_des=roll_des)
        rudder = self._beta_control(beta=beta, beta_des=0)
        return np.array([1 if f16.mach < 0.9 else 0, elevator, aileron, rudder])

    def _position_tracking(self, f16, pitch_des, yaw_des):
        roll = f16.euler[0] * rad2degree
        path_pitch = f16.path_pitch * rad2degree
        path_yaw = f16.path_yaw * rad2degree
        alpha = f16.alpha * rad2degree
        beta = f16.beta * rad2degree
        ground_speed = f16.ground_speed * feet2meter
        trajectory2earth = f16.rotation_trajectory2earth
        velocity2earth = f16.rotation_velocity2earth

        pitch_error = angle_error(angle=path_pitch, angle_des=pitch_des)
        yaw_error = angle_error(angle=path_yaw, angle_des=yaw_des)
        pitch_load = ground_speed * np.clip(0.02 * pitch_error, -1, 1) + 9.81 * np.cos(f16.path_pitch)
        yaw_load = ground_speed * np.clip(0.02 * yaw_error, -1, 1) * np.cos(f16.path_pitch)

        load_trajectory = np.array([0, yaw_load, -pitch_load])
        load_velocity = np.matmul(velocity2earth.T, np.matmul(trajectory2earth, load_trajectory))
        load = np.clip(-load_velocity[2] / 9.81, -1, 5)
        alpha_des = np.clip(load * 4, -4, 20)
        roll_des = np.arctan2(load_trajectory[1], -load_trajectory[2]) * rad2degree

        elevator = self._alpha_control(alpha=alpha, alpha_des=alpha_des)
        aileron = self._roll_control(roll=roll, roll_des=roll_des)
        rudder = self._beta_control(beta=beta, beta_des=0)
        return np.array([1 if f16.mach < 0.9 else 0, elevator, aileron, rudder])

    def _alpha_control(self, alpha, alpha_des):
        alpha_error = angle_error(angle=alpha, angle_des=alpha_des)
        alpha_velocity = angle_error(angle=self.alpha_last, angle_des=alpha) / self.time_step
        self.alpha_last = alpha
        self.alpha_integral = np.clip(self.alpha_integral + 10 * alpha_error * self.time_step, -2, 2)
        alpha_control = 0.8 * alpha_error - 1.8 * alpha_velocity + self.alpha_integral
        alpha_control = np.clip(alpha_control, -25, 25)
        return -alpha_control

    def _roll_control(self, roll, roll_des):
        roll_error = angle_error(angle=roll, angle_des=roll_des)
        roll_control = 0.07 * roll_error
        roll_control = np.clip(roll_control, -21.5, 21.5)
        return -roll_control

    def _beta_control(self, beta, beta_des):
        beta_error = angle_error(angle=beta, angle_des=beta_des)
        beta_velocity = (beta - self.beta_last) / self.time_step
        self.beta_last = beta
        beta_control = 12 * beta_error - 4 * beta_velocity
        beta_control = np.clip(beta_control, -30, 30)
        return beta_control
