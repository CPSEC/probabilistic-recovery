#!/usr/bin/env python
import os
from copy import deepcopy
from math import inf

import numpy as np
import rospy, rospkg
from lqr_control.msg import VehicleState
from lgsvl_msgs.msg import VehicleControlData

from utils.controllers.PID import PID
from utils.controllers.LQR import LQR
from utils.formal.gaussian_distribution import GaussianDistribution
from utils.formal.reachability import ReachableSet
from utils.formal.zonotope import Zonotope
from utils.observers import full_state_nonlinear_from_gaussian as fsn
from utils.info.Timer import Timer
from utils.observers.extended_kalman_filter import ExtendedKalmanFilter
from model import LaneKeeping
from kinematic_lateral_model import Kinematic
from sensor import Sensor, SensorData
from observer import Observer
from state_record import StateRecord
from settings import Settings


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


def main():
    control_rate = rospy.get_param("/control_frequency")
    dt = 1.0 / control_rate
    speed_P = rospy.get_param("/speed_P")
    speed_I = rospy.get_param("/speed_I")
    speed_D = rospy.get_param("/speed_D")
    speed_ref = rospy.get_param("/target_speed")
    attack_start_index = rospy.get_param("/attack_start_index")
    attack_end_index = rospy.get_param("/attack_end_index")
    attack_mode = rospy.get_param("/attack_mode")
    recovery_start_index = rospy.get_param("/recovery_start_index")
    debug = rospy.get_param("/debug")

    # get path file name 
    _rp = rospkg.RosPack()
    _rp_package_list = _rp.list()
    data_folder = os.path.join(_rp.get_path('recovery'), 'data')
    path_file = os.path.join(data_folder, 'cube_town_closed_line.txt')

    if debug:
        rospy.init_node('control_loop', log_level=rospy.DEBUG)
    else:
        rospy.init_node('control_loop', log_level=rospy.INFO)
    cmd = VehicleCMD()
    sensor = Sensor()
    observer = Observer(path_file, speed_ref)
    rec = StateRecord()
    gnd_rec = StateRecord()
    os.environ['bl'] = 'oprp_cl'
    rospy.on_shutdown(gnd_rec.save_data)
    timer = Timer()

    # speed PID controller
    speed_pid = PID(speed_P, speed_I, speed_D)
    speed_pid.setWindup(100)
    # steering LQR controller
    steer_model = LaneKeeping(speed_ref)
    Q = np.eye(4)
    R = np.eye(1) * 10
    steer_lqr = LQR(steer_model.A, steer_model.B, Q, R)
    steer_lqr.set_control_limit(np.array([-0.261799]), np.array([0.261799]))

    # Init for recovery
    recovery_complete_index = inf
    exp = Settings()
    model = Kinematic()
    U = Zonotope.from_box(exp.control_lo, exp.control_up)
    W = GaussianDistribution(np.zeros(model.n), np.eye(model.n)*1e-6)
    Q = W.sigma
    C_filter = np.array([[1, 0, 0], [0, 0, 1]]) if attack_mode == 0 else np.array([[1, 0, 0], [0, 1, 0]])
    R = np.zeros((C_filter.shape[0], C_filter.shape[0]))
    jh = lambda x, u: C_filter
    kalman_filter = ExtendedKalmanFilter(model.f, model.jfx, jh, Q, R)
    non_est = fsn.Estimator(model.ode, model.n, model.m, dt, W, kf=kalman_filter, jfx=model.jfx, jfu=model.jfu)
    

    rate = rospy.Rate(control_rate)
    time_index = 0  # time index
    while not rospy.is_shutdown():
        if sensor.ready:
            # cruise control
            if time_index >= recovery_complete_index:
                speed_pid.set_reference(0)
            else:
                speed_pid.set_reference(speed_ref)
            acc_cmd = speed_pid.update(sensor.data['v'])
            # model adaptation
            # v = state.velocity if state.velocity > 1 else 1
            # steer_model.update(v)
            # steer_lqr.update_gain(steer_model.A, steer_model.B, Q, R)

            sensor_ = SensorData()
            sensor_.data = deepcopy(sensor.data)
            # # sensor attack
            if attack_end_index > time_index >= attack_start_index:
                if attack_mode == 0:   # attack GPS sensor    y: -4
                    sensor_.data['y'] -= 4
                elif attack_mode == 1:  # attack IMU sensor    heading: -0.8
                    sensor_.data['yaw'] -= 0.8

            # record sensor data
            rec.record_x(sensor_.get_state(), time_index)

            timer.reset()
            if time_index < recovery_start_index:  # nominal controller
                feedback = observer.est(sensor_)
                rospy.logdebug(f"time_index={time_index}, e_d'={feedback[0]}, e_phi'={feedback[2]}, speed'={sensor.data['v']}")
                rospy.logdebug(f"     state={sensor.get_state()}")
                steer_target = steer_lqr.update(feedback)[0]
                
            if time_index == attack_start_index - 1:  # print ground truth
                rospy.logdebug(f"[trustworthy] i={time_index}, state={sensor.get_state()}")

            # state reconstruction & linearization
            if time_index == recovery_start_index:
                rospy.logdebug(f"[recovery start] i={time_index}, state={sensor.get_state()}")
                us = rec.get_us(attack_start_index - 1, time_index)
                x_0 = GaussianDistribution(rec.get_x(attack_start_index - 1), np.zeros((3, 3)))
                ys = rec.get_ys(C_filter, attack_start_index, time_index+1)
                x_cur, sysd = non_est.estimate(x_0, us, ys)
                rospy.logdebug(f"     recovered state: {x_cur.miu}")
            if recovery_start_index < time_index < recovery_complete_index:
                rospy.logdebug(f"[recoverying] i={time_index}, state={sensor.get_state()}")
                u_last =  rec.get_us(time_index-1, time_index)
                y_last = rec.get_ys(C_filter, time_index, time_index+1)
                x_cur, sysd = non_est.estimate(x_cur, u_last, y_last)
                rospy.logdebug(f"     predicted state: {x_cur.miu}")

            # call OPRP
            if recovery_start_index <= time_index < recovery_complete_index:
                reach = ReachableSet(sysd.A, sysd.B, U, W, max_step=100, c=sysd.c)
                reach.init(x_cur, exp.s)
                k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(max_k=exp.k_given)
                recovery_complete_index = time_index + k
                rec_u = U.alpha_to_control(alpha)
                steer_target = rec_u[0][0]

            # final step, stop
            if recovery_complete_index == time_index:
                rospy.loginfo(f"[recovery complete] i={time_index}, state={sensor.get_state()}")
                steer_target = 0

            timer.toc()

            # implement control input, consider control limit
            if steer_target > exp.control_up[0]:
                steer_target = exp.control_up[0]
            elif steer_target < exp.control_lo[0]:
                steer_target = exp.control_lo[0]
            cmd.send(acc_cmd, steer_target)
            rospy.logdebug(f"    control input={steer_target}")
            # record control input
            if time_index <= recovery_complete_index:
                rec.record_u(u=np.array([steer_target]), t=time_index)
                gnd_rec.record(x=sensor.get_state(), u=np.array([steer_target]), t=time_index)
                gnd_rec.record_ct(ct=timer.total(), t=time_index)

            time_index += 1
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass