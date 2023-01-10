from cmath import inf
from matrix_utils import hat, vee, q_to_R
from control import Control
from estimator import Estimator
from trajectory import Trajectory

import datetime
import time
import numpy as np
import pdb
import rospy
import threading
import copy

from geometry_msgs.msg import Pose, Twist, Wrench
from geometry_msgs.msg import Vector3, Point, Quaternion
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import Imu
from std_msgs.msg import String

# Rtss
from recovery.recovery_main_rtss import RecoveryRTSS
from recovery.recovery_main_emsoft import RecoveryEmsoft
from recovery.recovery_virtual_sensors import RecoveryVirtualSensor
from recovery.recovery_main_rtss_nonlinear import RecoveryRTSSNonlinear
RECOVERY_NAMES = ['rtss', 'emsoft', 'v_sensors', 'rtss_nonlinear']
class Rover:
    def __init__(self):

        self.on = True
        self.motor_on = False
        self.save_on = False
        self.mode = 0

        self.t0 = datetime.datetime.now()
        self.t = 0.0
        self.t_pre = 0.0
        self.freq_imu = 0.0
        self.freq_gps = 0.0
        self.freq_control = 0.0
        self.freq_log = 0.0

        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.a = np.zeros(3)
        self.R = np.identity(3)
        self.W = np.zeros(3)

        self.x_offset = np.zeros(3)
        self.yaw_offset = 0.0


        self.k_iter = 0
        

        self.g = 9.81
        self.ge3 = np.array([0.0, 0.0, self.g])

        # Gazebo uses ENU frame, but NED frame is used in FDCL.
        self.R_fg = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])

        self.V_R_imu = np.diag([0.01, 0.01, 0.01])
        self.V_x_gps = np.diag([0.01, 0.01, 0.01])
        self.V_v_gps = np.diag([0.01, 0.01, 0.01])
        self.max_sim = 600

        self.control = Control()
        self.control.use_integral = True  # Enable integral control

        self.attack_set = False

        self.estimator = Estimator()
        self.trajectory = Trajectory()

        self.lock = threading.Lock()
        self.t_init = time.time()

        self.freq = 0
        
    
    def init_recovery(self, freq, isolation, recovery_name, detection_delay, noise):
        self.freq = freq
        # Rtss
        self.recovery_complete_index = inf
        #
        self.detection_delay = detection_delay
        # self.k_attack = 15*freq
        self.k_attack = 40*freq
        self.k_detection = self.k_attack + detection_delay*freq
        self.isolation = isolation 
        self.k_max = self.k_detection + 10 * freq
        self.alarm = False
        self.recovered = False

        # attack sensor
        self.attack_sensor = int(2)
        self.attack_gps = False
        self.attack_size = -1

        #
        self.fM = np.zeros((4, 1))
        self.u_min = np.array([-2, -1e-3, -1e-3, -1e-3]) # Constraint the input so that the control action is close to the linearization point
        self.u_max = np.array([2, 1e-3, 1e-3, 1e-3]) # Constraint the input so that the control action is close to the linearization point

        file_name = "results/"
        self.estimated_states_vs = []
        self.recovery_name = recovery_name
        if recovery_name == RECOVERY_NAMES[0]:
            self.recovery = RecoveryRTSS(1/freq, self.u_min, self.u_max, self.attack_sensor, self.isolation, noise) 
            if self.isolation:
                file_name += "ours_cl_"
            else:
                file_name += "ours_"
        elif recovery_name == RECOVERY_NAMES[1]:
            self.recovery = RecoveryEmsoft(1/freq, self.u_min, self.u_max, self.attack_sensor, self.isolation, noise) 
            self.isolation = False
            file_name += "emsoft_"
        elif recovery_name == RECOVERY_NAMES[2]: 
            self.recovery = RecoveryVirtualSensor(1/freq, self.u_min, self.u_max, self.attack_sensor, self.isolation, noise) 
            self.isolation = False
            file_name += "virtual_sensors_"
        elif recovery_name == RECOVERY_NAMES[3]: 
            self.recovery = RecoveryRTSSNonlinear(1/freq, self.u_min, self.u_max, self.attack_sensor, self.isolation, noise) 
            if self.isolation:
                file_name += "ours_nonlinear_cl_"
            else:
                file_name += "ours_nonlinear_"
        else:
            raise NotImplemented('Input rtss, emsoft, v_sensor or nonlinear rtss')
        
        if self.isolation:
            n = self.recovery.system_model.n
            C = np.eye(n) 
            C = np.delete(C, (self.attack_sensor), axis=0)
            self.recovery.init_closed_loop(C)
        else:
            self.recovery.init_open_loop()

        # data store
        file_name += f'noise_{noise}'
        file_name_time = file_name + f'_time.txt'
        file_name_states = file_name + f'_states.txt'
        file_names_final_states = file_name + f'_final_states.txt'

        self.file_time = open(file_name_time, 'a')
        self.file_states = open(file_name_states, 'a')
        self.file_final_states = open(file_names_final_states, 'a')

        



    
    def update_current_time(self):
        t_now = datetime.datetime.now()
        self.t = (t_now - self.t0).total_seconds()


    def get_current_time(self):
        t_now = datetime.datetime.now()
        return (t_now - self.t0).total_seconds()
   
    # I should include the reconfiguration here!!
    # Implements the logic to switch between the nominal controller and the rtss controller
    def run_controller(self):
        self.update_current_time()
        with self.lock:
            t_init = time.time() - self.t_init

            self.states_pt = self.estimator.get_states()
            self.x, self.v, self.a, self.R, self.W = self.states_pt # copy the states for the gui
            states = copy.deepcopy(self.states_pt) 

            if states[0][0] > 5000.2 and not self.attack_set:
                self.k_attack = self.k_iter + 1
                self.k_detection = self.k_attack + 5
                self.attack_set = True
                self.k_max = self.k_detection + 10

            if self.k_iter == self.k_attack - 1:
                self.recovery.checkpoint_state( self.process_state(copy.deepcopy(states)) )
                # print("Checkpointed state:", self.process_state(copy.deepcopy(states)))

            # attack
            if self.k_iter >= self.k_attack and not self.attack_gps:
                states[0][self.attack_sensor] += self.attack_size
                pass

            desired = self.trajectory.get_desired(rover.mode, states, \
                self.x_offset, self.yaw_offset)
            # checkpoint ys and us
            # Gazebo uses a different frame
            u = copy.deepcopy(self.fM)
            u[2] = -u[2]
            u[3] = -u[3]
            if self.k_attack < self.k_iter <= self.k_detection:
                self.recovery.checkpoint(self.process_state(states), u)
            if self.recovery_name == RECOVERY_NAMES[0] or self.recovery_name == RECOVERY_NAMES[1] or self.recovery_name == RECOVERY_NAMES[3]:
                # compute reconfiguration for the first time
                if self.k_iter == self.k_detection:
                    self.alarm = True
                    fM, k_reconf_max = self.recovery.update_recovery_fi()
                    self.recovery_complete_index = self.k_iter + k_reconf_max
                    # print("reconfiguration begins", self.recovery.process_state(self.states_pt))
                elif self.k_detection < self.k_iter < self.recovery_complete_index:
                    fM, k_reconf = self.recovery.update_recovery_ni(self.process_state(states), u)
                    self.recovery_complete_index = self.k_iter + k_reconf
                    # print(fM)
                elif self.k_iter >= self.recovery_complete_index: # recovery finishes
                    fM = self.fM *0
                else: # nominal control 
                    fM = self.nominal_control(states, desired)
                if self.k_iter == self.recovery_complete_index or self.k_iter >= self.k_max:
                    # Store final state
                    self.write_final_states(self.states_pt)
                    self.k_iter = inf
                    # print('Number of steps: ', )
            elif self.recovery_name == RECOVERY_NAMES[2]: # TODO: Implement
                if self.k_detection == self.k_iter:
                    self.alarm = True
                    self.estimated_states_vs = self.recovery.update_recovery_fi()
                    fM = self.nominal_control(self.estimated_states_vs, desired)
                elif self.k_detection < self.k_iter <= self.k_max:
                    # Send the states that we estimated previous step and the control action
                    self.estimated_states_vs = self.recovery.update_recovery_ni(self.process_state(self.estimated_states_vs), u)
                    fM = self.nominal_control(self.estimated_states_vs, desired)
                elif self.k_iter > self.k_max:
                    fM = np.zeros((4, 1))
                else:
                    fM = self.nominal_control(states, desired)
                arrived = False
                if self.k_detection < self.k_iter <= self.k_max:
                    arrived = self.recovery.in_set(self.process_state(self.estimated_states_vs))
                if arrived or self.k_iter >= self.k_max:
                    # Store final state
                    self.write_final_states(self.states_pt)
                    self.k_iter = inf
            else:
                raise NotImplemented

            fM = self.saturate_input(fM)
            
            t_now = time.time() - self.t_init

            # Store time
            t = str(t_now - t_init)
            if self.k_iter < inf:
                self.file_time.write(f'{self.k_iter}, {self.k_detection}, {t}')
                self.file_time.write('\n')

            # Store trajectory
            state_print = self.print_state(self.process_state(self.states_pt))
            str_write = str(t_init) + "," + str(self.k_iter) + "," + str(self.detection_delay) + state_print
            self.file_states.write(str_write)
            self.file_states.write("\n")
            # print(t)



            self.fM = fM
            self.k_iter += 1
        return self.fM

    def write_final_states(self, states_pt):
        state_print = self.print_state(self.process_state(states_pt))
        str_write = str(self.k_iter) + "," + str(self.detection_delay) + state_print + str(self.k_iter - self.k_detection)
        self.file_final_states.write(str_write)
        self.file_final_states.write("\n")
        pass

    def print_state(self, states):
        text = ''
        for state in states:
            text = text + f', {state}'
        text = text + ","
        return text
    
    def process_state(self, x):
        pos = x[0]
        v = x[1]
        R = x[3]
        R = R.flatten()
        w = x[4]
        x = np.concatenate((pos, v, R, w)).flatten()
        return x

    def saturate_input(self, fM):
        fM = fM.flatten()
        min_f = 0
        max_f = 30
        min_t = -1
        max_t = 1
        fM[0] = np.clip(fM[0], min_f, max_f)
        for i in range(1, 4):
            fM[i] = np.clip(fM[i], min_t, max_t)
        return fM
    
    # Auxiliar function to call the nominal controller
    def nominal_control(self, states, desired):
        fM = self.control.run(states, desired)

        
        return fM


    def ros_imu_callback(self, message):
        q_gazebo = message.orientation
        a_gazebo = message.linear_acceleration
        W_gazebo = message.angular_velocity

        q = np.array([q_gazebo.x, q_gazebo.y, q_gazebo.z, q_gazebo.w])

        R_gi = q_to_R(q) # IMU to Gazebo frame
        R_fi = self.R_fg.dot(R_gi)  # IMU to FDCL frame (NED freme)

        # FDCL-UAV expects IMU accelerations without gravity.
        a_i = np.array([a_gazebo.x, a_gazebo.y, a_gazebo.z])
        a_i = R_gi.T.dot(R_gi.dot(a_i) - self.ge3)

        W_i = np.array([W_gazebo.x, W_gazebo.y, W_gazebo.z])

        with self.lock:
            self.estimator.prediction(a_i, W_i)
            self.estimator.imu_correction(R_fi, self.V_R_imu)


    def ros_gps_callback(self, message):
        x_gazebo = message.pose.pose.position
        v_gazebo = message.twist.twist.linear

        # Gazebo uses ENU frame, but NED frame is used in FDCL.
        if self.k_iter >= self.k_attack and self.attack_gps:
            x_g = np.array([x_gazebo.x, -x_gazebo.y, -x_gazebo.z - self.attack_size])
            v_g = np.array([v_gazebo.x, -v_gazebo.y, -v_gazebo.z])
        else:
            x_g = np.array([x_gazebo.x, -x_gazebo.y, -x_gazebo.z])
            v_g = np.array([v_gazebo.x, -v_gazebo.y, -v_gazebo.z])

        with self.lock:
            self.estimator.gps_correction(x_g, v_g, self.V_x_gps, self.V_v_gps)



def reset_uav():
    rospy.wait_for_service('/gazebo/set_model_state')
    
    init_position = Point(x=0.0, y=0.0, z=0.2)
    init_attitude = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    init_pose = Pose(position=init_position, orientation=init_attitude)

    zero_motion = Vector3(x=0.0, y=0.0, z=0.0)
    init_velocity = Twist(linear=zero_motion, angular=zero_motion)

    model_state = ModelState(model_name='uav', reference_frame='world', \
        pose=init_pose, twist=init_velocity)
    
    reset_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    reset_state(model_state)

    # print('Resetting UAV successful ..')


rover = Rover()
