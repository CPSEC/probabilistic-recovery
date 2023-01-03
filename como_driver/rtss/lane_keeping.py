#!/usr/bin/env python
'''
we subscribe to the steering angle topic /ecu/line_follow/str and publish to /ecu a steering and a velocity command.

This script also serves as a sample script for what other COMO higher-level control scripts can look like.

Author:
Sleiman Safaoui

Github:
The-SS

Email:
snsafaoui@gmail.com

Date:
July 30, 2018
'''

import roslib
import rospy
from barc.msg import ECU, Encoder
from como_image_processing.msg import LineData
from std_msgs.msg import Float64
import time
# Import numpy and scipy
from numpy import pi
import numpy as np
from scipy.signal import StateSpace


# Parameters To be moved to params file\
V_MAX = 15.0
V_MIN = -15.0
HEADING_MAX = np.pi/3
HEADING_MIN = -np.pi/3

# End params

# Controller
class PID_ctrl:
	'''
	Generic discrete time PID controller with integrator anti-windup and control output saturation
	'''

	def __init__(self, Kp = 1., Ki = 0., Kd = 0., i_max = 0., i_min = 0., ctrl_max = 1., ctrl_min = 0.):
		self.Kp = Kp
		self.Ki = Ki
		self.Kd = Kd
		self.i_max = i_max
		self.i_min = i_min
		self.ctrl_max = ctrl_max
		self.ctrl_min = ctrl_min
		self.e_curr = 0.
		self.e_prev = 0.
		self.e_sum = 0.
		self.t_curr = 0.
		self.t_prev = 0.

	def apply_pid(self, des_value, current_value, timestamp):
		self.e_curr = des_value - current_value
		#print('error', self.e_curr)
		self.t_curr = timestamp
		dt = self.t_curr - self.t_prev
		#print('dt', dt)

		# Proportional control
		p_ctrl = self.Kp * self.e_curr
		#print('p_ctrl', p_ctrl)

		# Integral control with anti-windup
		i_ctrl = self.e_sum + self.Ki*dt/2.0*(self.e_curr+self.e_prev)
		i_ctrl = min(i_ctrl, self.i_max)
		i_ctrl = max(i_ctrl, self.i_min)
		#print('i_ctrl', i_ctrl)

		# Derivative control
		d_ctrl = self.Kd*(self.e_curr-self.e_prev)/dt
		#print('d_ctrl', d_ctrl)

		# Control signal calculation
		ctrl = p_ctrl + i_ctrl + d_ctrl

		# Control saturation
		ctrl = min(ctrl, self.ctrl_max)
		ctrl = max(ctrl, self.ctrl_min)

		# update previous values with current values
		self.e_sum = i_ctrl
		self.e_prev = self.e_curr
		self.t_prev = self.t_curr

		return ctrl

# Sensors
class VelEst:
	'''
	Estimates the velocity of the car using the vel_est data
	'''
	def __init__(self):
		self.data_curr = []
		self.data_prev = []
		self.car_speed_est = []
	
	def update_data(self, data):
		if data == []:
			return
		self.data_prev = self.data_curr
		self.data_curr = data
		return
		
	def calc_vel (self):
		if (( self.data_curr == [] ) | ( self.data_prev == [] )):
			return 0.0
		vel_sum = 0.0
		num_elts = 0.0
		if self.data_curr.FL >= 0.00001:
			vel_sum += self.data_curr.FL
			num_elts += 1.
		if self.data_curr.FR >= 0.00001:
			vel_sum += self.data_curr.FR
			num_elts += 1.
		if self.data_curr.BL >= 0.00001:
			vel_sum += self.data_curr.BL
			num_elts += 1.
		if self.data_curr.BR >= 0.00001:
			vel_sum += self.data_curr.BR
			num_elts += 1.
		if self.data_prev.FL >= 0.00001:
			vel_sum += self.data_prev.FL
			num_elts += 1.
		if self.data_prev.FR >= 0.00001:
			vel_sum += self.data_prev.FR
			num_elts += 1.
		if self.data_prev.BL >= 0.00001:
			vel_sum += self.data_prev.BL
			num_elts += 1.
		if self.data_prev.BR >= 0.00001:
			vel_sum += self.data_prev.BR
			num_elts += 1.
		if num_elts > 0:
			vel = vel_sum/num_elts
			return vel
		else:
			return 0.0

class EncodersVelFetcher:
	def __init__(self):
		self.data_sub = rospy.Subscriber("/vel_est", Encoder, self.data_callback, queue_size =1)
		self.encoder_vel_data = []
		
	def data_callback(self, data):
		self.encoder_vel_data = data
		return
		
	def get_fetched_data(self):
		return self.encoder_vel_data

# Class to subscribe to the main controller
class CtrlSub:
	'''
	Subscribes to the main controller
	'''
	def __init__(self):
		self.str_cmd = ECU(0.0, 0.0)
		self.str_sub = rospy.Subscriber("/ecu/lane_keeping/", ECU, self.callback, queue_size =1) 
		self.begin = False
		
	def callback(self, data):
		self.str_cmd = data
		self.begin = True
		
	def get_str(self):
		return self.str_cmd.servo
	
	def get_motor(self):
		return self.str_cmd.motor
	
	def get_control(self):
		return self.str_cmd.motor, self.str_cmd.servo
		

# Class to publish the control commands to motor and steering
class ECUPub:
	'''
	Publishes an ECU command to /ecu topic
	'''
	def __init__(self):
		self.ecu = ECU(0.,0.)
		self.ecu_pub = rospy.Publisher('/ecu', ECU, queue_size = 1)
	
	def set_ecu(self, motor, servo):
		# Ensure saturation before publishing
		motor     = max(motor, V_MIN)
		motor     = min(motor, V_MAX)
		servo     = max(servo, HEADING_MIN)
		servo     = min(servo, HEADING_MAX) 
		# Dead zone correction
		servo = -float(servo) + np.pi/2
		servo = round(servo * 100)/100
		self.ecu = ECU(float(-motor - 8.5), servo) 
	
	def publish_ecu(self):
		self.ecu_pub.publish(self.ecu)


		
def main():
	# Initialize node for rtss
	rospy.init_node("lane_keeping_speed", anonymous=True)

	ecu_pub = ECUPub()
	vel_fetcher = EncodersVelFetcher()
	## Init things for low level control
	hlc_sub = CtrlSub()
	vel_est_fetcher = VelEst()
	vel_control = PID_ctrl(Kp = 15, Ki = 4, Kd = 0.4, \
							i_max = 40, i_min = -40, \
							ctrl_max = 15, ctrl_min = 0)
	rate = rospy.Rate(30)
	dt = 0.033333

	while not hlc_sub.begin:
		rate.sleep()
	timestamp = time.time()
	vel_control.t_prev = timestamp
	vel_control.t_curr = timestamp
	rate.sleep()
	while not rospy.is_shutdown():
		# Measure time
		timestamp = time.time()
		# Get current velocity
		vel_encoders = vel_fetcher.get_fetched_data()
		vel_est_fetcher.update_data(vel_encoders)
		current_velocity = vel_est_fetcher.calc_vel()
		# Get control command from high level control
		target_velocity, servo_cmd = hlc_sub.get_control()
		# apply PID
		mtr_cmd = vel_control.apply_pid(target_velocity, current_velocity, timestamp)
		if target_velocity <= 0:
			mtr_cmd = V_MIN
		# print("Current_velocity", round(current_velocity, 2), "target velocity", round(target_velocity, 2) , "motor_command: ", round(mtr_cmd, 2))
		# Set motor and servo command
		# print(target_velocity, mtr_cmd)
		ecu_pub.set_ecu(mtr_cmd, servo_cmd)
		# Publish
		ecu_pub.publish_ecu()
		rate.sleep()
		



if __name__ == "__main__":
	try:
		main()
	except rospy.ROSInterruptException:
		rospy.logfatal("ROS Interrupt. Shutting down line_follower_ctrl node")
		pass