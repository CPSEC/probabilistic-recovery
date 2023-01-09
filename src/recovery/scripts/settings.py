from math import pi, inf

import numpy as np

from utils.formal.strip import Strip

class Settings:
    control_lo = np.array([-30/360*pi])
    control_up = np.array([30/360*pi])
    target_set_lo = np.array([-55, 50.45-4.5, -0.2])
    target_set_up = np.array([55, 50.45-4, 0.2])
    k_given = 40
    output_index = 1
    s = Strip(np.array([0,1,0]),  a=target_set_lo[output_index], b=target_set_up[output_index])

    