#!/usr/bin/env python
import os

import numpy as np
import rospy, rospkg
from lqr_control.msg import VehicleState
from lgsvl_msgs.msg import VehicleControlData

from utils.controllers.PID import PID
from utils.controllers.LQR import LQR
from model import LaneKeeping
from sensor import Sensor
from observer import Observer


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
    control_rate = rospy.get_param("/control_frequency", 50)
    speed_P = rospy.get_param("/speed_P")
    speed_I = rospy.get_param("/speed_I")
    speed_D = rospy.get_param("/speed_D")
    speed_ref = rospy.get_param("/target_speed")
    attack_start_index = rospy.get_param("/attack_start_index")
    attack_end_index = rospy.get_param("/attack_end_index")
    attack_mode = rospy.get_param("/attack_mode")

    # get path file name 
    _rp = rospkg.RosPack()
    _rp_package_list = _rp.list()
    data_folder = os.path.join(_rp.get_path('recovery'), 'data')
    path_file = os.path.join(data_folder, 'cube_town_closed_line.txt')

    rospy.init_node('control_loop', log_level=rospy.DEBUG)
    # state = StateUpdate()
    cmd = VehicleCMD()
    sensor = Sensor()
    observer = Observer(path_file, speed_ref)

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
        if sensor.ready:
            # cruise control
            speed_pid.set_reference(speed_ref)
            acc_cmd = speed_pid.update(sensor.data['v'])
            # model adaptation
            # v = state.velocity if state.velocity > 1 else 1
            # steer_model.update(v)
            # steer_lqr.update_gain(steer_model.A, steer_model.B, Q, R)

            feedback = observer.est(sensor)
            rospy.logdebug(f"time_index={time_index}, e_d={feedback[0]}, e_phi={feedback[2]}, speed={sensor.data['v']}")
            rospy.logdebug(f"     x={sensor.data['x']}, y={sensor.data['y']}, theta={sensor.data['yaw']}")
            steer_target = steer_lqr.update(feedback)
            cmd.send(acc_cmd, steer_target)
            time_index += 1
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass