#!/usr/bin/env python
import numpy as np

import rospy
from lqr_control.msg import VehicleState
from lgsvl_msgs.msg import VehicleControlData

from utils.controllers.PID import PID
from utils.controllers.LQR import LQR
from model import LaneKeeping

class StateUpdate:
    def __init__(self) -> None:
        self.velocity = 0
        self.lateral_state = np.zeros((4,), dtype=np.float32)
        self.x = 0
        self.y = 0
        self.theta = 0

        self.started = False
        vehicle_state_topic = rospy.get_param("/vehicle_state_topic", "/vehicle_state")
        self.state_sub = rospy.Subscriber(vehicle_state_topic, VehicleState, self.callback)

    def callback(self, state):
        self.velocity = state.v
        self.lateral_state[0] = state.e_cg
        self.lateral_state[1] = state.e_cg_dot
        self.lateral_state[2] = state.theta_e
        self.lateral_state[3] = state.theta_e_dot
        self.x = state.x
        self.y = state.y
        self.theta = state.theta
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


def main():
    control_rate = rospy.get_param("/control_frequency")
    speed_P = rospy.get_param("/speed_P")
    speed_I = rospy.get_param("/speed_I")
    speed_D = rospy.get_param("/speed_D")
    speed_ref = rospy.get_param("/target_speed")
    attack_start_index = rospy.get_param("/attack_start_index")
    attack_end_index = rospy.get_param("/attack_end_index")
    attack_mode = rospy.get_param("/attack_mode")
    recovery_start_index = rospy.get_param("/recovery_start_index")

    rospy.init_node('control_loop', log_level=rospy.DEBUG)
    state = StateUpdate()
    cmd = VehicleCMD()

    # speed PID controller
    speed_pid = PID(speed_P, speed_I, speed_D)
    speed_pid.setWindup(100)
    # steering LQR controller
    steer_model = LaneKeeping(speed_ref)
    Q = np.eye(4)
    R = np.eye(1) * 10
    steer_lqr = LQR(steer_model.A, steer_model.B, Q, R)
    steer_lqr.set_control_limit(np.array([-0.261799]), np.array([0.261799]))
    
    rate = rospy.Rate(control_rate)
    time_index = 0  # time index
    while not rospy.is_shutdown():
        if state.started:
            # print(state.val.e_cg, state.val.e_cg_dot, state.val.theta_e, state.val.e_cg_dot)
            # cruise control
            speed_pid.set_reference(speed_ref)
            acc_cmd = speed_pid.update(state.velocity)
            # model adaptation
            # v = state.velocity if state.velocity > 1 else 1
            # steer_model.update(v)
            # steer_lqr.update_gain(steer_model.A, steer_model.B, Q, R)

            # sensor attack
            feedback = state.lateral_state.copy()
            if attack_end_index > time_index >= attack_start_index:
                if attack_mode == 0:   # attack GPS sensor
                    feedback += np.array([4, 0, 0, 0])
                elif attack_mode == 1:  # attack IMU sensor
                    feedback += np.array([0, 0, 1, 0])

            if time_index == attack_start_index - 1:  # print ground truth
                rospy.logdebug(f'[trustworthy] i={time_index}, x={state.x}, y={state.y}, theta={state.theta}')

            # state reconstruction & linearization
            if time_index == recovery_start_index:
                rospy.logdebug(f'[recovery] i={time_index}, x={state.x}, y={state.y}, theta={state.theta}')

            steer_target = steer_lqr.update(feedback)
            cmd.send(acc_cmd, steer_target)
            time_index += 1
            rospy.logdebug(f"time_index={time_index}, speed={state.velocity}, e_cg={state.lateral_state[0]}")
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass