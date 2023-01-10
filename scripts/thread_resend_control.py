from rover import rover

import datetime
import numpy as np
import rospy

from geometry_msgs.msg import Wrench
from geometry_msgs.msg import Vector3

def thread_resend_control(noise_variance):
    # print('CONTROL: thread starting ..')

    pub = rospy.Publisher('uav_fm', Wrench, queue_size=1)

    freq = 100
    rate = rospy.Rate(freq) # 200 hz

    freq = 0.0
    t = datetime.datetime.now()
    t_pre = datetime.datetime.now()
    avg_number = 1000
    
    counter = 10
    noise = np.zeros((4, 1))
    while rover.freq == 0:
        rate.sleep()
    while rover.k_iter < rover.k_max and rover.on:
        counter += 1
        t = datetime.datetime.now()
        dt = (t - t_pre).total_seconds()
        if dt < 1e-6:
            continue
        
        freq = (freq * (avg_number - 1) + (1 / dt)) / avg_number
        t_pre = t
        # rover.freq_control = freq

        fM = rover.fM
        fM_send = np.zeros((4, 1))
        fM_send[0] = fM[0]
        if counter >= freq /rover.freq:
            counter = 0
            for i in range (0, 4):
                noise[i] = np.random.normal(0, noise_variance)

        for i in range (1, 4):
            fM_send[i] = fM[i] + noise[i]
        
        if (not rover.motor_on) or (rover.mode < 2):
            fM_message = Wrench(force=Vector3(x=0.0, y=0.0, z=0.0), \
                torque=Vector3(x=0.0, y=0.0, z=0.0))
        else:
            fM_message = Wrench(force=Vector3(x=0.0, y=0.0, z=fM_send[0]), \
                torque=Vector3(x=fM_send[1], y=fM_send[2], z=fM_send[3]))
        # print(fM_message)
        pub.publish(fM_message)

        rate.sleep()
    
    # print('CONTROL: thread closed!')