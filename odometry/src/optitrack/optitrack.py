#!/usr/bin/env python
# Import ros 
import rospy
import roslib
from std_msgs.msg import String
from como_image_processing.msg import LineData
from std_msgs.msg import Float64
from barc.msg import Encoder, ECU
# Import useful libraries
import socket
import numpy as np
import time
# to be moved to a param file

class UDPCommunication():
	def __init__(self, port):
		self._socket = []
		self.address = socket.gethostname()
		self.port = port
		
	def create_server(self):
		self._socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		self._socket.bind(("", self.port))

	def receive_info(self):
		data, addr = self._socket.recvfrom(1024)
		data = data.decode('utf-8')
		return data
	
	def close(self):
		self._socket.close()
		pass

class Interpreter():
	def __init__(self):
		self.x     = 0
		self.y 	   = 0
		self.theta = 0
		self.time  = -1
	
	# Assume message t;x;y;theta
	def get_message(self, message):
		try:
			msg = message.split(";")
			time = float( msg[0] )
			if time > self.time:
				self.x = float( msg[1] )
				self.y = float( msg[2] )
				self.theta = float( msg[3] )
				self.time = time
		except:
			print("Error reading optitrack cameras")
		return self.x, self.y, self.theta

class Optitrack():
	def __init__(self):
		self.state = np.array([0.0, 0.0, 0.0])

		self.optitrack_pub_x = rospy.Publisher("/optitrack_x", Float64, queue_size =1)
		self.optitrack_pub_y = rospy.Publisher("/optitrack_y", Float64, queue_size =1)
		self.optitrack_pub_t = rospy.Publisher("/optitrack_t", Float64, queue_size =1)
	
    
	def set_values(self, x, y, theta):
		self.state[0] = x
		self.state[1] = y
		self.state[2] = theta
	
	def publish_camera(self):
		self.optitrack_pub_x.publish(self.state[0])  
		self.optitrack_pub_y.publish(self.state[1])  
		self.optitrack_pub_t.publish(self.state[2])  

def main():
	rospy.init_node("optitrack")
	rate = rospy.Rate(50)
	
	port = 12345
	udp_socket = UDPCommunication(port)
	udp_socket.create_server()
	interperter = Interpreter()
	publisher = Optitrack()
	
	# Initial position
	x = np.array([0, 0, 0])
	print("Ready to receive info from the camera!!")
	time_init = time.time()
	try:
		while not rospy.is_shutdown():
			message = udp_socket.receive_info()
			x, y, theta = interperter.get_message(message)
			publisher.set_values(x, y, theta)
			publisher.publish_camera()

			time_current = time.time()
			# print(f'{time_current - time_init}, {x}, {y}, {theta}')
			rate.sleep()
	except e:
		print(e)
		
		



		
		

	

if __name__=='__main__':
	main()



		

