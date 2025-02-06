import numpy as np


"""unit translation"""
feet2meter = 0.3048
meter2feet = 1/feet2meter
rad2degree = 180/np.pi
degree2rad = np.pi/180

"""F16 parameter"""
rho0 = 2.377e-3                     # see-level air density
xcg = 0.35                          # center of gravity position
xcgr = 0.35                         # reference center of gravity position
s = 300                             # wing area
b = 30                              # wing span
cbar = 11.32                        # mean aerodynamic chord
mass = 20500 / 32.17049             # mass, in 20500 lb, approximate 637.229958 slug
he = 160.0                          # angular momentum for engine
gravity = np.array([0, 0, 32.17])   # gravity ft/s2 = 9.805 m/s2

Jxx = 9496                          # rotational inertia, in slug
Jyy = 55814
Jzz = 63100
Jxz = 982
J = np.array([[Jxx, 0, -Jxz],
              [0, Jyy, 0],
              [-Jxz, 0, Jzz]])
Jinv = np.linalg.inv(J)


def angle_error(angle, angle_des):
    """normalized angle error between -180 ~ 180"""
    angle_e = angle_des - angle
    if angle_e > 180:
        angle_e = angle_e - 360
    elif angle_e < -180:
        angle_e = angle_e + 360
    return angle_e


earth_radius = 6378137      # in meter


def xyz2llh(position):
    x = position[0]
    y = position[1]
    h = -position[2]

    latitude = x / (earth_radius + h) * rad2degree + 0.5
    longitude = y / (earth_radius + h) * rad2degree + 0.5
    return [longitude, latitude, h]