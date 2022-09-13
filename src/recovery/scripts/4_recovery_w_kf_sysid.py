#!/usr/bin/env python
import os
import numpy as np
from scipy.signal import StateSpace

import rospy
import rospkg
from lqr_control.msg import VehicleState
from lgsvl_msgs.msg import VehicleControlData
from gui.msg import AttackCmd

from utils.controllers.PID import PID
from utils.controllers.LQR import LQR
from model import LaneKeeping
from utils.formal.zonotope import Zonotope
from utils.formal.gaussian_distribution import GaussianDistribution
from utils.formal.reachability import ReachableSet
from utils.formal.strip import Strip
from utils.observers.kalman_filter import KalmanFilter


class StateUpdate:
    def __init__(self) -> None:
        self.velocity = 0
        self.lateral_state = np.zeros((4,), dtype=np.float32)

        self.started = False
        vehicle_state_topic = rospy.get_param("/vehicle_state_topic", "/vehicle_state")
        self.state_sub = rospy.Subscriber(vehicle_state_topic, VehicleState, self.callback)

    def callback(self, state):
        self.velocity = state.v
        self.lateral_state[0] = state.e_cg
        self.lateral_state[1] = state.e_cg_dot
        self.lateral_state[2] = state.theta_e
        self.lateral_state[3] = state.theta_e_dot
        self.started = True


class VehicleCMD:
    def __init__(self) -> None:
        vehicle_cmd_topic = rospy.get_param("/vehicle_cmd_topic", "/vehicle_cmd")
        self.cmd_pub = rospy.Publisher(vehicle_cmd_topic, VehicleControlData, queue_size=1000)

    def send(self, acc_cmd, steer_target):
        control_cmd = VehicleControlData()
        control_cmd.header.stamp = rospy.Time.now()
        control_cmd.acceleration_pct = acc_cmd
        control_cmd.target_gear =  VehicleControlData.GEAR_DRIVE
        control_cmd.target_wheel_angle = steer_target
        self.cmd_pub.publish(control_cmd)

class AttackCMD:
    def __init__(self) -> None:
        self.cmd_sub = rospy.Subscriber("attack_cmd", AttackCmd, self.callback)
        self.triggered = False

    def isAttacked(self):
        return self.triggered

    def reset(self):
        self.triggered = False

    def callback(self, cmd):
        self.triggered = True
        self.bias = np.array([cmd.b0, cmd.b1, cmd.b2, cmd.b3])
        bias_e = [cmd.b0e, cmd.b1e, cmd.b2e, cmd.b3e]
        C_filter = []
        for i in range(4):
            if bias_e[i]:
                continue
            else:
                tmp = [0, 0, 0, 0]
                tmp[i] = 1
                C_filter.append(tmp)
        self.C_filter = np.array(C_filter)
        self.attack_duration = cmd.attack_duration
        self.detection_delay = cmd.detection_delay

    def read_cmd(self):
        self.reset()
        return self.bias, self.C_filter, self.attack_duration, self.detection_delay


class SysId:
    def __init__(self, filename) -> None:
        self.filepath = filename
        self.x = []
        self.u = []
    def record(self, x, u):
        self.x.append(x.copy())
        self.u.append(u.copy())
    def compute(self):
        length = len(self.x)
        x_next = np.array(self.x[1:]).astype(np.float)
        x = np.array(self.x[:-1])
        u = np.array(self.u[:-1])
        x_u = np.concatenate((x, u), axis=1).astype(np.float)
        print(f'{x.shape=},{u.shape=}')
        A_B = np.linalg.lstsq(x_u, x_next, rcond=-1)[0].T
        Ad = A_B[:,:4]
        Bd = A_B[:,4:]
        print(f'{Ad=},\n{Bd=}')
        with open(self.filepath, 'wb') as f:
            np.savez(f, Ad=Ad, Bd=Bd)
        rospy.logdebug('interrupt by keyboard')


def main():
    control_rate = rospy.get_param("/control_frequency", 50)
    control_interval = 1.0/control_rate
    speed_P = rospy.get_param("/speed_P")
    speed_I = rospy.get_param("/speed_I")
    speed_D = rospy.get_param("/speed_D")
    speed_ref = rospy.get_param("/target_speed")
    # parameters for recovery
    # attack_start_index = rospy.get_param("/attack_start_index")
    # attack_end_index = rospy.get_param("/attack_end_index")
    # attack_mode = rospy.get_param("/attack_mode")
    # recovery_start_index = rospy.get_param("/recovery_start_index")
    attack_start_index = 10000
    attack_end_index = 10000
    recovery_start_index = 10000
    max_recovery_step = rospy.get_param("/max_recovery_step")
    recovery_end_index = 10000 # to be computed
    # store the system model
    _rp = rospkg.RosPack()
    _rp_package_list = _rp.list()
    data_folder = os.path.join(_rp.get_path('recovery'), 'data')
    model_file = os.path.join(data_folder, 'model.npz')
    record_start = False


    rospy.init_node('control_loop', log_level=rospy.DEBUG)
    state = StateUpdate()
    cmd = VehicleCMD()
    attack = AttackCMD()
    sysid = SysId(model_file)
    rospy.on_shutdown(sysid.compute)

    # speed PID controller
    speed_pid = PID(speed_P, speed_I, speed_D)
    speed_pid.setWindup(100)
    # steering LQR controller
    steer_model = LaneKeeping(speed_ref)
    sysc = StateSpace(steer_model.A, steer_model.B, steer_model.C, steer_model.D)
    sysd = sysc.to_discrete(control_interval)
    Q = np.eye(4)
    R = np.eye(1) * 10
    steer_lqr = LQR(sysc.A, sysc.B, Q, R)
    control_lo = np.array([-0.261799])
    control_up = np.array([0.261799])
    steer_lqr.set_control_limit(control_lo, control_up)
    # recovery
    U = Zonotope.from_box(control_lo, control_up)
    C_noise = np.diag([0.01, 0.001, 0.001, 0.001])
    W = GaussianDistribution.from_standard(np.zeros((4,))*0.01, C_noise)
    safe_set = Strip(np.array([1, 0, 0, 0]), a=-0.05, b=0.05)
    x_cur_update = None
    last_steer_target = None
    recovery_control_sequence = None
    bias = np.zeros((4,))
    # pre-process for recovery
    reach = ReachableSet(sysd.A, sysd.B, U, W, max_step=max_recovery_step+2)
    kf = KalmanFilter(sysd.A, sysd.B, sysd.C, sysd.D, Q, R)

    
    rate = rospy.Rate(control_rate)
    time_index = 0  # time index
    while not rospy.is_shutdown():
        if state.started:
            # info 
            rospy.logdebug(f"i={time_index}, v={state.velocity:.2f}, e_cg={state.lateral_state[0]:.2f}")
            if time_index == attack_start_index:
                rospy.loginfo('-'*10+' Attack Start '+'-'*10)
            if time_index == attack_end_index - 1:
                rospy.loginfo('-'*10+' Attack End '+'-'*10)
            if time_index == recovery_start_index:
                rospy.loginfo('='*10+' Recovery Start '+'='*10)
            if time_index == recovery_end_index - 1:
                rospy.loginfo('='*10+' Recovery End '+'='*10)
            # print(state.val.e_cg, state.val.e_cg_dot, state.val.theta_e, state.val.e_cg_dot)

            # cruise control
            speed_pid.set_reference(speed_ref)
            acc_cmd = speed_pid.update(state.velocity)
            # model adaptation
            # v = state.velocity if state.velocity > 1 else 1
            # steer_model.update(v)
            # steer_lqr.update_gain(steer_model.A, steer_model.B, Q, R)

            # process attack command
            if attack.isAttacked():
                record_start = True
                bias, C_filter, attack_duration, detection_delay = attack.read_cmd()
                attack_start_index = time_index
                attack_end_index = time_index + attack_duration
                recovery_start_index = time_index + detection_delay
                kf = KalmanFilter(sysd.A, sysd.B, C_filter, sysd.D, Q, R)
                rospy.loginfo(f"[ATTACK]  {bias=}, {C_filter=}, {attack_duration=}, {detection_delay=}")

            # sensor attack
            feedback = state.lateral_state.copy()
            if attack_start_index <= time_index < attack_end_index:
                feedback += bias
            steer_target = steer_lqr.update(feedback)

            if time_index == recovery_start_index:
                # state reconstruction here
                reconstructed_state = state.lateral_state.copy()
                x_cur_update = GaussianDistribution(reconstructed_state, np.eye(4)*0.01)
                reach.init(x_cur_update, safe_set)

                print(f"x_0={x_cur_update.miu=}")
                k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(max_k=max_recovery_step)
                print(f"{k=}, {z_star=}, {P=}")
                recovery_end_index = recovery_start_index + k
                # attack_end_index = recovery_end_index - 1  #   for test!!!
                recovery_control_sequence = U.alpha_to_control(alpha)
                print('recovery_control=', recovery_control_sequence[0,:])

            if recovery_start_index < time_index < recovery_end_index:
                x_cur_predict = GaussianDistribution(*kf.predict(x_cur_update.miu, x_cur_update.sigma, last_steer_target))
                y = C_filter @ feedback
                x_cur_update = GaussianDistribution(*kf.update(x_cur_predict.miu, x_cur_predict.sigma, y))
                reach.init(x_cur_update, safe_set)

                print(f"x_0={x_cur_update.miu=}")
                k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(max_k=max_recovery_step)
                print(f"{k=}, {z_star=}, {P=}")
                recovery_end_index = time_index + k
                # attack_end_index = recovery_end_index - 1  #   for test!!!
                recovery_control_sequence = U.alpha_to_control(alpha)
                print('recovery_control=', recovery_control_sequence[0,:])

            if recovery_start_index <= time_index < recovery_end_index:
                i = time_index - recovery_start_index
                steer_target = recovery_control_sequence[0][0]
                rospy.logdebug(f"i={time_index}, steer_target={steer_target:.4f}, e_cg={state.lateral_state[0]:.2f}")

            last_steer_target = np.array([steer_target])
            cmd.send(acc_cmd, steer_target)
            time_index += 1

            # record data for system identification
            if record_start:
                sysid.record(x=state.lateral_state, u=np.array([steer_target]))
            
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass



"""
All road:
x.shape=(1987, 4),u.shape=(1987, 1)
Ad=array([[ 2.02835245e+00,  1.12616305e+01, -2.83601013e+01,
        -2.16484717e+01],
       [ 3.80120533e-02,  9.07168835e-02, -2.26161646e-01,
        -1.72619013e-01],
       [ 9.01166233e-03, -2.26612746e-01,  5.75942878e-01,
         4.39687199e-01],
       [ 7.78601058e-03, -1.72946519e-01,  4.39644601e-01,
         3.35634981e-01]]),
Bd=array([[ 2.77555756e-15],
       [ 3.38271078e-17],
       [-4.51028104e-17],
       [-1.21430643e-16]])

Only straight / with left turn:
x.shape=(252, 4),u.shape=(252, 1)
Ad=array([[ 0.99116342, -0.68664772, -0.79449042,  1.22542675],
       [ 0.13423463,  0.3021741 ,  0.33829842, -0.27942423],
       [ 0.13381132,  0.33848106,  0.37927469, -0.3205379 ],
       [ 0.25302769, -0.27998292, -0.32095334,  0.43082868]]),
Bd=array([[-1.30104261e-16],
       [ 6.07153217e-18],
       [ 2.68882139e-17],
       [-1.30104261e-17]])
"""