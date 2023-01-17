import numpy as np
import time
import os
import rospkg, rospy
import csv
from settings import Settings

class StateRecord:
    def __init__(self) -> None:
        self.xs = []
        self.us = []
        self.ts = []
        self.cts = []
        self.start = time.time()
        self.exp = Settings()
    
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

    def record_ct(self, ct, t):
        assert t == len(self.cts)
        self.cts.append(ct)

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
        self.save_all_times()
        self.save_final_state()
        
    
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

    def save_all_times(self):
        _rp = rospkg.RosPack()
        _rp_package_list = _rp.list()
        data_folder = os.path.join(_rp.get_path('recovery'), 'data')
        file_name = os.path.join(data_folder, 'svl', 'time_'+os.environ['bl']+'.csv')

        if not os.path.exists(file_name):
            with open(file_name, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['k', 'time', 'comp_time'])
        with open(file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            cnt = len(self.cts)
            for i in range(cnt):
                writer.writerow([i, self.ts[i], self.cts[i]])

    @staticmethod
    def in_target_set(target_lo, target_hi, x_cur):
        res = True
        for i in range(len(x_cur)):
            if target_lo[i] > x_cur[i] or target_hi[i] < x_cur[i]:
                res = False
                break
        return res

    def save_final_state(self):
        _rp = rospkg.RosPack()
        _rp_package_list = _rp.list()
        data_folder = os.path.join(_rp.get_path('recovery'), 'data')
        file_name = os.path.join(data_folder, 'svl', 'final_states.csv')

        if not os.path.exists(file_name):
            with open(file_name, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['name', 'k', 'time', 'steps_recovery', 'attack_sz', 'x', 'y', 'yaw', 'Success'])
        with open(file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            # prepare data
            name = os.environ['bl']
            k = len(self.xs)
            time = self.ts[-1]
            recovery_start_index = rospy.get_param("/recovery_start_index")
            steps_recovery = k - recovery_start_index
            attack_sz = 0
            x = self.xs[-1][0]
            y = self.xs[-1][1]
            yaw = self.xs[-1][2]
            success = 1 if self.in_target_set(self.exp.target_set_lo, self.exp.target_set_up, self.xs[-1]) else 0
            data = [name, k, time, steps_recovery, attack_sz, x, y, yaw, success]
            writer.writerow(data)

            



