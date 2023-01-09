from math import sqrt, atan, tan, cos, sin

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
from kinematic_lateral_model import l_f, l_r, delta_r

class SensorData:
    def __init__(self) -> None:
        self.data = {'x':0, 'y':0, 'yaw':0}

    def get_state(self):
        return np.array([self.data['x'], self.data['y'], self.data['yaw']])

class Sensor(SensorData):
    def __init__(self) -> None:
        """
        x, y:               position
        v:                  velocity
        roll, pitch, yaw:   Euler  (yaw->heading)
        angular_v:          z axis angular velocity
        a:                  acceleration
        """
        super().__init__()
        self.ready = False
        self.odom_ready = False
        self.imu_ready = False
        vehicle_odom_topic = rospy.get_param("/vehicle_odom_topic")
        vehicle_imu_topic = rospy.get_param("vehicle_imu_topic")
        self.odom_sub = rospy.Subscriber(vehicle_odom_topic, Odometry, self.odom_callback)
        self.imu_sub = rospy.Subscriber(vehicle_imu_topic, Imu, self.imu_callback)

    def odom_callback(self, odom: Odometry):
        self.data['x'] = odom.pose.pose.position.x
        self.data['y'] = odom.pose.pose.position.y
        self.data['v'] = ((odom.twist.twist.linear.x)**2 + (odom.twist.twist.linear.y)**2)**0.5

        if self.odom_ready == False:
            self.odom_ready = True
            if self.imu_ready == True:
                self.ready = True

    def imu_callback(self, imu: Imu):
        q = imu.orientation
        q_lst = [q.x, q.y, q.z, q.w]
        self.data['roll'], self.data['pitch'], self.data['yaw'] = euler_from_quaternion(q_lst)
        self.data['angular_v'] = imu.angular_velocity.z
        self.data['a'] = sqrt(imu.linear_acceleration.x**2 + imu.linear_acceleration.y**2)

        if self.imu_ready == False:
            self.imu_ready = True
            if self.odom_ready == True:
                self.ready = True





