from math import pi, inf

import numpy as np

from utils.formal.strip import Strip

class Settings:
    control_lo = np.array([-30/360*pi])
    control_up = np.array([30/360*pi])
    target_set_lo = np.array([-55, 50.45-4.5, -pi])
    target_set_up = np.array([55, 50.45-3, pi])
    safe_set_lo = np.array([-55, 50.45-8, -pi])
    safe_set_up = np.array([55, 60, pi])
    recovery_ref = (target_set_lo + target_set_up)/2
    Q =  np.diag([0, 1, 1])
    QN = np.diag([0, 1, 1])
    R = np.eye(1)
    k_given = 40
    output_index = 1
    s = Strip(np.array([0,1,0]),  a=target_set_lo[output_index], b=target_set_up[output_index])

    