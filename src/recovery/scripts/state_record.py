import numpy as np
import time
import os
import rospkg, rospy
import csv

class StateRecord:
    def __init__(self) -> None:
        self.xs = []
        self.us = []
        self.ts = []
        self.start = time.time()
    
    def record(self, x, u, t):
        assert t == len(self.xs)
        self.xs.append(x)
        self.us.append(u)
        self.ts.append(time.time() - self.start)

    def record_x(self, x, t):
        assert t == len(self.xs)
        self.xs.append(x)
        self.ts.append(time.time() - self.start)

    def record_u(self, u, t):
        assert t == len(self.us)
        self.us.append(u)

    def get_x(self, i):
        assert i < len(self.xs)
        return self.xs[i]

    def get_xs(self, start, end):
        return self.xs[start:end]

    def get_us(self, start, end):
        return self.us[start:end]

    def get_ys(self, C, start, end):
        xs = np.vstack(self.xs[start:end])
        ys = (C @ xs.T).T
        return ys


    def save_data(self):
        save_state = rospy.get_param("/save_state")
        if save_state:
            self.save_all_states()
    
    def save_all_states(self):
        # filename
        _rp = rospkg.RosPack()
        _rp_package_list = _rp.list()
        data_folder = os.path.join(_rp.get_path('recovery'), 'data')
        file_name = os.path.join(data_folder, 'svl', 'states_'+os.environ['bl']+'.csv')

        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['k', 'time', 'x', 'y', 'yaw'])
            cnt = len(self.xs)
            for i in range(cnt):
                writer.writerow([i, self.ts[i]] + list(self.xs[i]))