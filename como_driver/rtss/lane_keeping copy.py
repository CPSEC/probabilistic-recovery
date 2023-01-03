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
# Import Rtss
from utils.formal.gaussian_distribution import GaussianDistribution
from utils.formal.reachability import ReachableSet
from utils.formal.zonotope import Zonotope
from utils.observers.kalman_filter import KalmanFilter
from utils.formal.strip import Strip

# Parameters To be moved to params file
K_ATTACK = 65# Time attack begins. To be moved to a param file
K_DETECTION = 65 + 2# Time attack begins. To be moved to a param file
K_MAX = 50

V_MAX = 1
V_MIN = 0
HEADING_MAX = np.pi/3
HEADING_MIN = -np.pi/3
ENCODER_SPEED_2_REAL = 1
L = 0.32
ISOLATION = False
lr = 0.15
lf = 0.17

x_linear = [0, 0, 0]
u_linear = [0.6, 0]
# End params


# not sure if to move this to the main control file.
class RTSS:
	# Inputs
	# A, B, C, D: system matrices
	# W: Noise matrix
	# u_min, u_max: minimum and maximum control
	# recovery_index: recovery begins
	def __init__(self, Ad, Bd, Cd, Dd, W, u_min, u_max, recovery_index):
		self.Ad = Ad
		self.Bd = Bd
		self.Cd = Cd
		self.Dd = Dd
		# Create zonotope
		self.U = Zonotope.from_box(u_min, u_max)
		# Create reachable set
		self.reach = ReachableSet(Ad, Bd, self.U, W, max_step=K_MAX + 10)
		# Create strip
		self.s = Strip(np.array([0, 1, 0.1]), a=-0.6, b=-0.4)
		# self.s = Strip(np.array([0, 1, 0]), a=-0.5, b=-0.3)
		# Attack detection
		self.recovery_index = recovery_index
		# Create 
		self.kf = None
		self.x_cur_update = None
	
	def set_kalman_filter(self, C, Q, R):
		self.kf = KalmanFilter(self.Ad, self.Bd, C, self.Dd, Q, R)
	
	def recovery_isolation_fs(self, us, ys, x_0):
		us = np.array(us)
		ys = np.array(ys)
		x_res, P_res = self.kf.multi_steps(x_0, np.zeros_like(self.Ad), us, ys)
		x_cur_update = GaussianDistribution(x_res[-1], P_res[-1])
		self.x_cur_update = x_cur_update
		self.reach.init(x_cur_update, self.s)
		k, X_k, D_k, z_star, alpha, P, arrive = self.reach.given_k(max_k=K_MAX)
		rec_u_temp = self.U.alpha_to_control(alpha)
		return rec_u_temp[0], k
	
	def recovery_isolation_ns(self, cur_x, cur_u):
		cur_u = np.array(cur_u).reshape( (len(self.Bd[0]), ) )
		cur_x = np.array(cur_x).reshape( (len(self.Ad[0]), ) )
		x_cur_predict = GaussianDistribution(*self.kf.predict(self.x_cur_update.miu, self.x_cur_update.sigma, cur_u))
		y = self.kf.C @ cur_x
		# print("current state:", cur_x)
		# print("current meas:", y)
		# print("update: ", self.x_cur_update)
		# print("predict: ", x_cur_predict)
		x_cur_update = GaussianDistribution(*self.kf.update(x_cur_predict.miu, x_cur_predict.sigma, y))
		self.x_cur_update = x_cur_update
		self.reach.init(x_cur_update, self.s)
		k, X_k, D_k, z_star, alpha, P, arrive = self.reach.given_k(max_k=K_MAX)
		rec_u_temp = self.U.alpha_to_control(alpha)
		return rec_u_temp[0], k
		
	
	def recovery_no_isolation(self, us, x_0):
		us = np.array(us)
		n = len(self.Ad)
		x_0 = GaussianDistribution( x_0, np.zeros((n, n)) )
		self.reach.init( x_0, self.s )
		self.x_res_point = self.reach.state_reconstruction(us)
		self.reach.init( self.x_res_point, self.s )
		# Run one-step rtss
		k, X_k, D_k, z_star, alpha, P, arrive = self.reach.given_k(max_k=K_MAX)
		# Compute u
		rec_u = self.U.alpha_to_control(alpha)
		return rec_u, k
	
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
	

class System():
	def __init__(self, dt):
		
		model = ["rear", "cg"]
		model = "rear"
		if model == "rear":
			filter = 2
			self.A = [[0, 0, -u_linear[0] * np.sin(x_linear[2])],
				[0, 0,  u_linear[0] * np.cos(x_linear[2])],
				[0, 0, -1/filter*0]]
			self.B = [[np.cos(x_linear[2]), 0],
				[np.sin(x_linear[2]), 0],
				[np.tan(u_linear[1])/L, (u_linear[0] * (np.tan(u_linear[1])**2 + 1) / L) / 1]]
			self.C = [[1, 0, 0],
				[0, 1, 0],
				[0, 0, 1]]
			self.D = [[0, 0], [0, 0], [0, 0]]
		else:
			c1 = lr**2 * np.tan(u_linear[1])**2 / L**2 + 1 # \frac{\tan^2(\delta)}{L^2} + 1
			c2 = np.tan(u_linear[1])**2 + 1 # \tan(\delta)^2/L^2 + 1
			c3 = np.sin( x_linear[2] + np.arctan( lr*np.tan(u_linear[1]) / L ) ) # \sin( \theta + arctan( lr\frac{tan(\delta)}{L} ) )
			c4 = np.cos( x_linear[2] + np.arctan( lr*np.tan(u_linear[1]) / L ) ) # \cos( \theta + arctan( lr\frac{\delta}{L} ) )
			self.A = [[0, 0, -u_linear[0]*c3], # [0, 0, -v sin( \theta + \atan( \frac{tan(\delta)}{L} ) )]
					[0, 0, u_linear[0]*c4],
					[0, 0, 0]]
			self.B = [[ c4, -lr * u_linear[0] * c3 * c2 / ( L*c1 ) ], 
			[c3, lr * u_linear[0]*c4*c2 / ( L* c1 )], 
			[np.tan(u_linear[1]) / ( L*(c1**(1/2)) ), u_linear[0]*c2/( L*(c1**(1/2)) ) - lr**2 * u_linear[0] * np.tan(u_linear[1])**2 * c2 / (L**3 * c1**(3/2))]]
			self.C = [[1, 0, 0],
				[0, 1, 0],
				[0, 0, 1]]
			self.D = [[0, 0], [0, 0], [0, 0]]

		self.B = [[dynamic[1]] for dynamic in self.B ] # Fed just the angle
		self.D = [[d[1]] for d in self.D ] # Fed just the angle
		print(self.A)
		print(self.B)
		print(self.D)

		self.dt = dt
		# self.u_min = np.array( [V_MIN, HEADING_MIN] )
		# self.u_max = np.array( [V_MAX, HEADING_MAX] )
		self.u_min = np.array( [HEADING_MIN] )
		self.u_max = np.array( [HEADING_MAX] )
		sysc = StateSpace(self.A, self.B, self.C, self.D)
		sysd = sysc.to_discrete(self.dt)
		self.Ad = sysd.A
		self.Bd = sysd.B
		self.Cd = sysd.C
		self.Dd = sysd.D
		

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



# Class to publish the control commands to motor and steering
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

	# Init rtss
	n = 3 
	covariance = np.eye( n ) * 1/100000 
	# covariance[2, 2] = 1
	mu = np.zeros((n,))
	W = GaussianDistribution.from_standard(mu, covariance)
	
	system = System(dt)
	rtss = RTSS(system.Ad, system.Bd, system.Cd, system.Dd, W, system.u_min, system.u_max, K_DETECTION)
	if ISOLATION:
		C_kf = np.array([[1, 0, 0], [0, 0, 1]])
		Q = W.sigma 
		R = np.eye(C_kf.shape[0]) *0
		rtss.set_kalman_filter(C_kf, Q, R)
	
	# Init instant
	k = 0
	us = []
	ys = []
	recovery_complete_index = 100000000

	
	while not position_fetcher.begin:
		rate.sleep()
	timestamp = time.time()
	lateral_ctrl.t_prev = timestamp
	lateral_ctrl.t_curr = timestamp
	while not rospy.is_shutdown():
		t_begin = time.time()
		# Fetch current state 
		timestamp = time.time()
		state = position_fetcher.get_odometer_info()
		# print("Position from camera: ", state)
		# print(f'x: {state[0]}, y: {state[1]}, theta: {state[2]}')
		if k == K_ATTACK: # Checkpoint state. attack begins
			print("Attack Begins")
			store_state = state.copy()
			# print("state attack: ", state)
		if k >= K_ATTACK:
			state[1] -= 0.3
			

		# Checkpoint ys and us (measurements and inputs)
		if K_ATTACK <= k <= K_DETECTION: # Time between attack begins and attack is detected
			u = np.array([str_cmd])
			us.append(u)
			if ISOLATION:
				ys.append( C_kf @ state )

		if k == K_DETECTION: # attack controller
			print("Reconfiguration begins")
			
			print("Reconfiguration begins!!: ", position_fetcher.get_odometer_info())
			if ISOLATION: #  recovery with isolation
				str_cmd, k_reconf_max = rtss.recovery_isolation_fs(us, ys, store_state)
				recovery_complete_index = k + k_reconf_max
				print("reconstructed state: ", rtss.x_cur_update)
			else: # recovery without isolation
				u_recovery, k_reconf_max = rtss.recovery_no_isolation(us, store_state)
				k_reconf = 0
				recovery_complete_index = k_reconf_max + k
				str_cmd = u_recovery[k_reconf]
				print("reconstructed state:", rtss.x_res_point)
				# print(u_recovery)
		# print(k, recovery_complete_index)
		mtr_cmd = u_linear[0] #get motor command
		if K_DETECTION < k < recovery_complete_index:			
			if ISOLATION: #  recovery with isolation
				u = np.array([str_cmd])
				str_cmd, k_reconf_max = rtss.recovery_isolation_ns( state, u)
				recovery_complete_index = k + k_reconf_max
				pass
			else: # Recovery without isolation
				k_reconf += 1
				if k_reconf < k_reconf_max: 
					str_cmd = u_recovery[k_reconf]
		elif k > recovery_complete_index: # Once the reconfiguration finishes, stop
			# print("Reconfiguration finished!!: ", state, "strip: ", rtss.s.l @ state)
			mtr_cmd = -200
			str_cmd = 0	
			# pass
		else: # Fetch nominal controller
			servo = lateral_ctrl.apply_pid(0, state[1], timestamp)
			str_cmd = servo #get steering command
			mtr_cmd = u_linear[0] #get motor command

			# str_cmd = str_sub.get_str() #get steering command
			# mtr_cmd = str_sub.get_motor() #get motor command
		if k == recovery_complete_index:
			state = position_fetcher.get_odometer_info()
			print("Reconfiguration finished!!: ", state[0], ", ", state[1], ", ", state[2], ", ", "strip: ", rtss.s.l @ state)
		# Publishes control commands 
		ecu_publisher.set_ecu(float(mtr_cmd * ENCODER_SPEED_2_REAL), float(str_cmd))
		ecu_publisher.publish_ecu()
		

		k += 1
		t_end = time.time()

		# print("Main loop: ", t_end - t_begin)
		rate.sleep()
		



if __name__ == "__main__":
	try:
		main()
	except rospy.ROSInterruptException:
		rospy.logfatal("ROS Interrupt. Shutting down line_follower_ctrl node")
		pass