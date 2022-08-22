#!/usr/bin/env python
import rospy
import roslib
import numpy as np
from std_msgs.msg import String
from como_image_processing.msg import LineData
from std_msgs.msg import Float64
from barc.msg import Encoder, ECU

# to be moved to a param file
L = 0.33
lr = 0.15
lf = 0.17


class StrSub:
	'''
	Subscribes to the steering angle topic
	'''
	def __init__(self):
		self.str_cmd = ECU(0, 0)
		self.str_sub = rospy.Subscriber("/ecu", ECU, self.callback, queue_size =1)
		
	def callback(self, data):
		self.str_cmd = data
		
	def get_str(self):
		# print(self.str_cmd.servo)
		# Convert angle.
		# Zero becomes go straight. negative angle going right
		return -(self.str_cmd.servo - np.pi/2)

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
	 
class Odometer():
	def __init__(self):
		self.state = np.array([0.0, 0.0, 0.0])

		self.odometer_pub_x = rospy.Publisher("/odometry_x", Float64, queue_size =1)
		self.odometer_pub_y = rospy.Publisher("/odometry_y", Float64, queue_size =1)
		self.odometer_pub_t = rospy.Publisher("/odometry_t", Float64, queue_size =1)
	
    
	def update_position(self, velocity, delta, dt):
		# Gains found by experimentation
		self.state[0] += velocity * np.cos( self.state[2] ) * dt 
		self.state[1] += velocity * np.sin( self.state[2] ) * dt 
		self.state[2] += velocity * np.tan( delta ) * dt / L * 0.5 # 0.225
		
		# beta = np.arctan( lr/(lr + lf) * np.tan(delta) )
		# self.state[0] += velocity * np.cos( self.state[2] + beta ) * dt * 0.38
		# self.state[1] += velocity * np.sin( self.state[2] + beta ) * dt * 0.38
		# self.state[2] += velocity / lr * np.sin(beta) * dt * 0.265
		

		# print("states odometry:", self.state)
		# return self.state
	
	def publish_odometer(self):
		self.odometer_pub_x.publish(self.state[0])  
		self.odometer_pub_y.publish(self.state[1])  
		self.odometer_pub_t.publish(self.state[2])  

		

def main():
	rospy.init_node("odometry")
	rate = rospy.Rate(30)
	dt = 0.033333
	vel_est_fetcher = VelEstFetcher()
	car_vel_est = CarVelEst()
	str_sub = StrSub()
	odometer = Odometer()
	
	# Initial position
	x = np.array([0, 0, 0])
	str_cmd_store = []
	while not rospy.is_shutdown():
		vel_est_encoders = vel_est_fetcher.get_fetched_data() # get speed v
		car_vel_est.update_data(vel_est_encoders)
		car_vel = car_vel_est.calc_vel()

		str_cmd = str_sub.get_str() # get steering command \delta
		str_cmd_store += [str_cmd]
		if len(str_cmd_store) > 15:
			str_cmd_store.pop(0)
		str_cmd = np.mean(str_cmd_store)

		# print("angle: ", str_cmd)
		odometer.update_position(car_vel, str_cmd, dt)


		print(f'{odometer.state[0]}, {odometer.state[1]}, {odometer.state[2]}')
		odometer.publish_odometer()
		
		rate.sleep()



		
		

	

if __name__=='__main__':
	main()



          

