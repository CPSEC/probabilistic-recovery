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
from cmath import inf
# Import Recovery
from recovery.recovery_main_rtss import RecoveryRTSS
from recovery.recovery_main_emsoft import RecoveryEmsoft
from recovery.recovery_virtual_sensors import RecoveryVirtualSensor
from recovery.recovery_main_rtss_nonlinear import RecoveryRTSSNonlinear

# Parameters To be moved to params file
K_ATTACK = 65# Time attack begins. To be moved to a param file
K_DETECTION = 65 + 1# Time attack begins. To be moved to a param file
K_MAX = K_DETECTION + 50

V_MAX = 1
V_MIN = 0
HEADING_MAX = np.array( [np.pi/3] )
HEADING_MIN = np.array( [-np.pi/3] )
ENCODER_SPEED_2_REAL = 1.22

ATTACKED_SENSOR = int(0)
ATTACK_SIZE = -0.4
RECOVERY_NAMES = ['rtss', 'emsoft', 'v_sensors', 'rtss_nonlinear']
recovery_name = RECOVERY_NAMES[3]
ISOLATION = True

x_linear = [0, 0, 0]
u_linear = [0.6, 0]


ref_after_attack = np.array([-0.5, 0])
# End params

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

		return np.array(ctrl)
	

		

# Class that retrieves info from the odometry
class OdometryFetcher():
	# Fetch x, y and theta
	def __init__(self):
		self.odometer_sub_x = rospy.Subscriber("/odometry_x", Float64, self.callback_x, queue_size =1)
		self.odometer_sub_y = rospy.Subscriber("/odometry_y", Float64, self.callback_y, queue_size =1)
		self.odometer_sub_t = rospy.Subscriber("/odometry_t", Float64, self.callback_t, queue_size =1)
		self.states = np.array([0.0, 0.0, 0.0])
		self.x = 0.0
		self.y = 0.0
		self.t = 0.0
		self.begin = False
	# Function to fetch x
	def callback_x(self, data):
		self.x = data.data
		self.begin = True
	# Function to fetch y
	def callback_y(self, data):
		self.y = data.data
		self.begin = True
	# Function to fetch theta
	def callback_t(self, data):
		self.t = data.data
		self.begin = True
	
	# Retrieves odometry information
	def get_odometer_info(self):
		self.states[0] = self.x
		self.states[1] = self.y
		self.states[2] = self.t
		return self.states


# Class that retrieves info from the optitrack system
class OptitrackFetcher():
	# Fetch x, y and theta
	def __init__(self):
		self.odometer_sub_x = rospy.Subscriber("/optitrack_x", Float64, self.callback_x, queue_size =1)
		self.odometer_sub_y = rospy.Subscriber("/optitrack_y", Float64, self.callback_y, queue_size =1)
		self.odometer_sub_t = rospy.Subscriber("/optitrack_t", Float64, self.callback_t, queue_size =1)
		self.states = np.array([0.0, 0.0, 0.0])
		self.x = 0.0
		self.y = 0.0
		self.t = 0.0
		self.begin = False
	# Function to fetch x
	def callback_x(self, data):
		self.x = data.data
		self.begin = True
	# Function to fetch y
	def callback_y(self, data):
		self.y = data.data
		self.begin = True
	# Function to fetch theta
	def callback_t(self, data):
		self.t = data.data
		self.begin = True
	
	# Retrieves odometry information
	def get_odometer_info(self):
		self.states[0] = self.x
		self.states[1] = self.y
		self.states[2] = self.t
		return self.states



# Class to publish the control commands to low level controller
class ECUPub:
	'''
	Publishes an ECU command to /ecu topic
	'''
	def __init__(self):
		self.ecu = ECU(0.,-100.)
		self.ecu_pub = rospy.Publisher('/ecu/lane_keeping/', ECU, queue_size = 1)
	
	def set_ecu(self, motor, servo):
		# Ensure saturation before publishing
		# motor     = max(motor, V_MIN - 2)
		motor     = min(motor, V_MAX)
		servo     = max(servo, HEADING_MIN)
		servo     = min(servo, HEADING_MAX) 
		# Dead zone correction
		servo = float(servo) 
		self.ecu = ECU(motor, servo) 
	
	def publish_ecu(self):
		self.ecu_pub.publish(self.ecu)


		
def main():
	# Initialize node for rtss
	rospy.init_node("lane_keeping", anonymous=True)

	# Instantiates main controller fetcher
	# str_sub = CtrlSub()
	# Instantiates ecu publisher
	ecu_publisher = ECUPub()
	# Instantiate odometer fetcher
	position_fetcher = OptitrackFetcher()
	# position_fetcher = OdometryFetcher()

	lateral_ctrl_Kp= 12 # 8
	lateral_ctrl_Ki= 0.1
	lateral_ctrl_Kd= 3 # 1
	lateral_ctrl = PID_ctrl(Kp = lateral_ctrl_Kp, Ki = lateral_ctrl_Ki, Kd = lateral_ctrl_Kd, \
							i_max = HEADING_MAX, i_min = HEADING_MIN, \
							ctrl_max = HEADING_MAX, ctrl_min = HEADING_MIN)
	# Rate 
	rate = rospy.Rate(10)
	dt = 0.1

	f_name = '/home/odroid/como/workspace/src/como_driver/launch/results/'
	if recovery_name == RECOVERY_NAMES[0]:
		recovery = RecoveryRTSS(ref_after_attack, dt, HEADING_MIN, HEADING_MAX, ISOLATION)
		iso = ISOLATION
		f_name += "ours_"
	elif recovery_name == RECOVERY_NAMES[1]:
		recovery = RecoveryEmsoft(ref_after_attack, dt, HEADING_MIN, HEADING_MAX)
		iso = False
		f_name += "emsoft_"
	elif recovery_name == RECOVERY_NAMES[2]:
		recovery = RecoveryVirtualSensor(ref_after_attack, dt, HEADING_MIN, HEADING_MAX)
		iso = False
		f_name += "vs_"
	elif recovery_name == RECOVERY_NAMES[3]:
		recovery = RecoveryRTSSNonlinear(ref_after_attack, dt, HEADING_MIN, HEADING_MAX, ISOLATION)
		iso = ISOLATION
		f_name += "ours_nonlinear_"
	else:
		raise NotImplemented

	if iso:
		n = recovery.system_model.n
		C = np.eye(n)
		C = np.delete(C, (ATTACKED_SENSOR), axis=0)
		recovery.init_closed_loop(C)
		f_name += "cl_"
		print("Closed loop")
	else:
		f_name += "ol_"
		recovery.init_open_loop()
		print("Open loop")
	
	# Init instant
	k = 0
	recovery_complete_index = 100000000

	
	while not position_fetcher.begin:
		rate.sleep()
	timestamp = time.time()
	lateral_ctrl.t_prev = timestamp
	lateral_ctrl.t_curr = timestamp
	mtr_cmd = u_linear[0] #get motor command
	arrived = False


	file_final_states = open(f_name + "final_states.txt", 'a')
	file_final_states.write("k,time,steps_recovery,attack_sz,final_pos_x,final_pos_y,final_angle")
	file_final_states.write('\n')

	file_time = open(f_name + "time.txt", 'a')
	file_time.write("k,time,comp_time")
	file_time.write('\n')

	file_states = open(f_name + "states.txt", 'a')
	file_states.write("k,time,pos_x,pos_y,angle")
	file_states.write('\n')

	t0 = time.time()
	while not rospy.is_shutdown() and k <= K_MAX + 1:
		t_begin = time.time() - t0
		# Fetch current state 
		timestamp = time.time()
		state = position_fetcher.get_odometer_info()
		state = np.array(state[1:])
		# print("Actual state:    ", state)
		# print("Position from camera: ", state)
		# print(f'x: {state[0]}, y: {state[1]}, theta: {state[2]}')
		# Implement attack
		if k >= K_ATTACK:
			state[ATTACKED_SENSOR] += ATTACK_SIZE
		# Checkpoint state
		if k == K_ATTACK - 1:
			recovery.checkpoint_state(state.copy())
			print(state)
		# Checkpoint for closed loop
		if K_ATTACK < k <= K_DETECTION:
			recovery.checkpoint(state, str_cmd)

		if recovery_name == RECOVERY_NAMES[0] or recovery_name == RECOVERY_NAMES[1] or recovery_name == RECOVERY_NAMES[3]: # If ours or emsoft
			if k == K_DETECTION: # if the attack is detected
				str_cmd, k_reconf_max = recovery.update_recovery_fi()
				recovery_complete_index = k + k_reconf_max
			elif K_DETECTION < k < recovery_complete_index: # During the recovery
				str_cmd, k_reconf = recovery.update_recovery_ni(state, str_cmd)
				recovery_complete_index = k + k_reconf
			elif k >= recovery_complete_index: # I recovery finishes. Stop vehicle and store data
				str_cmd *= 0
				mtr_cmd *= 0
				# Save final state
				r_state = position_fetcher.get_odometer_info()
				file_final_states.write(f'{k},{t_begin},{k - K_DETECTION},{ATTACK_SIZE},{r_state[0]},{r_state[1]},{r_state[2]}')
				file_final_states.write('\n')
				k = inf
			else: # otherwise (before attack detected) nominal control
				str_cmd = lateral_ctrl.apply_pid(0, state[0], timestamp)
		elif recovery_name == RECOVERY_NAMES[2]: # if virtual sensors
			if K_DETECTION == k: # Attack detected
				estimated_states = recovery.update_recovery_fi()
				str_cmd = lateral_ctrl.apply_pid(ref_after_attack[0], estimated_states[0], timestamp)
			elif K_DETECTION < k < K_MAX: # Reconfiguration
				estimated_states = recovery.update_recovery_ni(estimated_states, str_cmd)
				str_cmd = lateral_ctrl.apply_pid(ref_after_attack[0], estimated_states[0], timestamp)
			elif k > K_MAX: # Stop if the system has not recovered after some time
				str_cmd *= 0
				mtr_cmd *= 0
				# Store data
				r_state = position_fetcher.get_odometer_info()
				file_final_states.write(f'{k},{t_begin},{k - K_DETECTION},{ATTACK_SIZE},{r_state[0]},{r_state[1]},{r_state[2]}')
				file_final_states.write('\n')
				print("Final state", r_state[0])
			else: # Nominal controller
				str_cmd = lateral_ctrl.apply_pid(0, state[0], timestamp)
			if K_DETECTION <= k <= K_MAX: # If attack detected
				arrived = recovery.in_set(estimated_states) 
				if arrived: # if arrived to the set, stop and store data
					str_cmd *= 0
					mtr_cmd *= 0
					# Store data
					r_state = position_fetcher.get_odometer_info()
					file_final_states.write(f'{k},{t_begin},{k - K_DETECTION},{ATTACK_SIZE},{r_state[0]},{r_state[1]},{r_state[2]}')
					file_final_states.write('\n')
					k = inf
					print("Final state", r_state)
		else:
			raise NotImplemented
		
		# Publishes control commands 
		ecu_publisher.set_ecu(float(mtr_cmd * ENCODER_SPEED_2_REAL), float(str_cmd))
		ecu_publisher.publish_ecu()
		u = str_cmd

		k += 1
		t_end = time.time() - t0
		# print(t_end - t_begin, ", ", state[0], ", ", state[1], ", ", str_cmd)
		# print("Main loop: ", t_end - t_begin)

		# Save data
		st = position_fetcher.get_odometer_info()
		file_states.write(f'{k},{t_end},{st[0]},{st[1]},{st[2]}')
		file_states.write('\n')
		if not k == inf:
			file_time.write(f'{k},{t_end},{t_end - t_begin}')
			file_time.write('\n')
		
		rate.sleep()
	file_final_states.close()
	file_time.close()
	file_states.close()
	print("Finished")
		



if __name__ == "__main__":
	try:
		main()
	except rospy.ROSInterruptException:
		rospy.logfatal("ROS Interrupt. Shutting down line_follower_ctrl node")
		pass