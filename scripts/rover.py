from cmath import inf
from matrix_utils import hat, vee, q_to_R
from control import Control
from estimator import Estimator
from trajectory import Trajectory

import datetime
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
from recovery_main_rtss import RecoveryRTSS
from recovery_main_emsoft import RecoveryEmsoft
from recovery_virtual_sensors import RecoveryVirtualSensor

RECOVERY_NAMES = ['rtss', 'emsoft', 'v_sensors']
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

        self.control = Control()
        self.control.use_integral = True  # Enable integral control

        self.attack_set = False

        self.estimator = Estimator()
        self.trajectory = Trajectory()

        self.lock = threading.Lock()
        
    
    def init_recovery(self, freq, isolation, recovery_name):
        self.freq = freq
        # Rtss
        self.recovery_complete_index = inf
        self.k_iter = 0
        # 
        self.k_attack = 1000000
        self.k_detection = self.k_attack + freq
        self.isolation = isolation 
        self.k_max = self.k_detection + 10

        # attack sensor
        self.attack_sensor = int(2)
        self.attack_gps = False

        #
        self.fM = np.zeros((4, 1))
        self.u_min = np.array([-2, -1e-3, -1e-3, -1e-3])
        self.u_max = np.array([2, 1e-3, 1e-3, 1e-3])


        self.states_recovery = []
        self.recovery_name = recovery_name
        if recovery_name == RECOVERY_NAMES[0]:
            self.recovery = RecoveryRTSS(1/freq, self.u_min, self.u_max, self.attack_sensor, self.isolation) 
        elif recovery_name == RECOVERY_NAMES[1]:
            self.recovery = RecoveryEmsoft(1/freq, self.u_min, self.u_max, self.attack_sensor, self.isolation) 
            self.isolation = False
        elif recovery_name == RECOVERY_NAMES[2]: # TODO: implement
            self.recovery = RecoveryVirtualSensor(1/freq, self.u_min, self.u_max, self.attack_sensor, self.isolation) 
            self.isolation = False
        else:
            raise NotImplemented('Input rtss, emsoft or v_sensor')
        
        if self.isolation:
            n = self.recovery.system_model.n
            C = np.eye(n) 
            C = np.delete(C, (self.attack_sensor), axis=0)
            self.recovery.init_closed_loop(C)
        else:
            self.recovery.init_open_loop()

    
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
            t_init = datetime.datetime.now()

            states_pt = self.estimator.get_states()
            self.x, self.v, self.a, self.R, self.W = states_pt # copy the states for the gui
            states = copy.deepcopy(states_pt) # FIXME: this is a deep copy of the estimator to avoid attacking the real estimator... 
                                              # This is just a proof of concept for now

            if states[0][0] > 5.2 and not self.attack_set:
                self.k_attack = self.k_iter + 1
                self.k_detection = self.k_attack + 2 * self.freq
                self.attack_set = True
                self.k_max = self.k_detection + 10

            if self.k_iter == self.k_attack - 1:
                self.recovery.checkpoint_state(copy.deepcopy(states))
                print("Checkpointed state:", states)

            # attack
            if self.k_iter >= self.k_attack and not self.attack_gps:
                states[0][self.attack_sensor] += -0.5
                pass
            # checkpoint ys and us
            # Gazebo uses a different frame
            u = copy.deepcopy(self.fM)
            u[2] = -u[2]
            u[3] = -u[3]
            if self.k_attack < self.k_iter <= self.k_detection:
                self.recovery.checkpoint(states, u)
            if self.recovery_name == RECOVERY_NAMES[0] or self.recovery_name == RECOVERY_NAMES[1]:
                # compute reconfiguration for the first time
                if self.k_iter == self.k_detection:
                    fM, k_reconf_max = self.recovery.update_recovery_fi()
                    self.recovery_complete_index = self.k_iter + k_reconf_max
                    print("reconfiguration begins", self.recovery.process_state(states_pt))
                elif self.k_detection < self.k_iter < self.recovery_complete_index:
                    fM, k_reconf = self.recovery.update_recovery_ni(states, u)
                    self.recovery_complete_index = self.k_iter + k_reconf
                elif self.k_iter >= self.recovery_complete_index: # recovery finishes
                    fM = self.fM *0
                else: # nominal control 
                    fM = self.nominal_control(states)
                if self.k_iter == self.recovery_complete_index:
                    print("recovery finished, ", states_pt)
            elif self.recovery_name == RECOVERY_NAMES[2]: # TODO: Implement
                if self.k_detection == self.k_iter:
                    states = self.recovery.update_recovery_fi()
                    self.states_recovery = states
                    fM = self.nominal_control(states)
                elif self.k_detection < self.k_iter <= self.k_max:
                    states = self.recovery.update_recovery_ni(self.states_recovery, u)
                    fM = self.nominal_control(states)
                elif self.k_iter > self.k_max:
                    fM = np.zeros((4, 1))
                else:
                    fM = self.nominal_control(states)
                if self.k_iter == self.k_max + 1:
                    print("reconfiguration finishes", self.recovery.process_state(states_pt))


            
            if self.k_iter < self.recovery_complete_index:
                # print(fM)
                t_now = datetime.datetime.now()
                t = (t_now - t_init).total_seconds()
                # print(t)



            self.fM = fM
            self.k_iter += 1
        return self.fM
    
    # Auxiliar function to call the nominal controller
    def nominal_control(self, states):
        desired = self.trajectory.get_desired(rover.mode, states, \
            self.x_offset, self.yaw_offset)
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
            x_g = np.array([x_gazebo.x, -x_gazebo.y, -x_gazebo.z - 0.4])
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

    print('Resetting UAV successful ..')


rover = Rover()
