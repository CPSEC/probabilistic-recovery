#!/usr/bin/env python
import rospy
import roslib
import numpy as np
from std_msgs.msg import String
from como_image_processing.msg import LineData
from std_msgs.msg import Float64
from barc.msg import Encoder, ECU

TARGET_VELOCITY = 0.7

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


class VelEstFetcher:
	def __init__(self):
		self.data_sub = rospy.Subscriber("/vel_est", Encoder, self.data_callback, queue_size =1)
		self.encoder_vel_data = []
		
	def data_callback(self, data):
		self.encoder_vel_data = data
		return
		
	def get_fetched_data(self):
		return self.encoder_vel_data

		
class CarVelEst: # Maybe I need to move this to a different file 
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
		if ((self.data_curr == []) | (self.data_prev == [])):
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


class VelPub():
	def __init__(self):
		self.speed = 0.0

		self.pub_speed = rospy.Publisher("/speed", Float64, queue_size =1)
	
    
	def set_speed(self, speed):
		# Gains found by experimentation
        self.speed = speed
		
		
		
	
	def publish_speed(self):
		self.pub_speed.publish(self.speed)  


		
def main():
    # Initialize node for rtss
	rospy.init_node("speed_control", anonymous=True)

	encoder_velocity_fetcher = VelEstFetcher()
    velocity_estimator = CarVelEst()
    velocity_publisher = VelPub()

	velocity_ctrl_Kp= 1
	velocity_ctrl_Ki= 0.010
	velocity_ctrl_Kd= 0.010
	velocity_ctrl = PID_ctrl(Kp = velocity_ctrl_Kp, Ki = velocity_ctrl_Ki, Kd = velocity_ctrl_Kd, \
							i_max = V_MAX, i_min = V_MIN, \
							ctrl_max = V_MAX, ctrl_min = V_MIN)
    # Rate 
	rate = rospy.Rate(10)
	dt = 0.1

	
	timestamp = time.time()
	velocity_ctrl.t_prev = timestamp
	velocity_ctrl.t_curr = timestamp
	while not rospy.is_shutdown():
        vel_est_encoders = encoder_velocity_fetcher.get_fetched_data()
        velocity_estimator.update_data(vel_est_encoders)
		car_vel = velocity_estimator.calc_vel()

        timestamp = time.time()
        mtr_cmd = velocity_ctrl.apply_pid(TARGET_VELOCITY, car_vel, timestamp)

        velocity_publisher.set_speed( mtr_cmd )

        velocity_publisher.publish()

		rate.sleep()
		



if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.logfatal("ROS Interrupt. Shutting down speed_ctrl node")
        pass