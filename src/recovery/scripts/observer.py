from math import cos, sin, pi, tan

import numpy as np
import rospy
from path import Path
from sensor import Sensor

class Observer:
    def __init__(self, path_file, r_speed) -> None:
        self.path = Path(path_file)
        self.r_speed = r_speed

    def est(self, s: Sensor):
        ri = self.path.get_nearest_index(s.data['x'], s.data['y'])
        dx = self.path.x[ri] - s.data['x']
        dy = self.path.y[ri] - s.data['y']
        theta_m = self.path.heading[ri]
        cos_theta_m = cos(theta_m)
        sin_theta_m = sin(theta_m) 
        e_d = cos_theta_m * dy - sin_theta_m * dx
        phi = s.data['yaw']
        e_phi = self.normalize_angle(theta_m-phi)
        e_d_dot = s.data['v'] * tan(e_phi)
        e_phi_dot = s.data['angular_v'] - self.path.kappa[ri] * self.r_speed
        return np.array([e_d, e_d_dot, e_phi, e_phi_dot])

    # NormalizeAngle
    @staticmethod
    def normalize_angle(angle):
        a = (angle + pi) % (2 * pi)
        if a < 0.0: 
            a += (2.0 * pi)
        return a - pi


    