from rover import rover

import datetime
import numpy as np
import rospy

from geometry_msgs.msg import Wrench
from geometry_msgs.msg import Vector3

from cmath import inf


def thread_control(detection_delay, reconfiguration, noise, isolation):
    # print('CONTROL: thread starting ..')

    # pub = rospy.Publisher('uav_fm', Wrench, queue_size=1)

    # freq = 20
    freq = 50
    rate = rospy.Rate(freq) # 200 hz
    recovery_name = ['rtss', 'emsoft', 'v_sensors', 'rtss_nonlinear']
    if 0 <= reconfiguration <= 3:
        recovery_name = recovery_name[reconfiguration]
    else:
        raise NotImplemented
    if isolation == 1:
        iso = True
    else:
        iso = False
    rover.init_recovery(freq=freq, isolation=iso, recovery_name = recovery_name, detection_delay=detection_delay, noise=noise)

    # freq = 0.0
    t = datetime.datetime.now()
    t_pre = datetime.datetime.now()
    avg_number = 100

    while rover.k_iter <= rover.k_max and rover.on:
        t = datetime.datetime.now()
        dt = (t - t_pre).total_seconds()
        if dt < 1e-6:
            continue
        
        freq = (freq * (avg_number - 1) + (1 / dt)) / avg_number
        t_pre = t
        rover.freq_control = freq
        
        try:
            fM = rover.run_controller()
        except:
            rover.write_final_states(rover.states_pt)
            rover.k_iter = inf
            print("error")
        
        # if (not rover.motor_on) or (rover.mode < 2):
        #     fM_message = Wrench(force=Vector3(x=0.0, y=0.0, z=0.0), \
        #         torque=Vector3(x=0.0, y=0.0, z=0.0))
        # else:
        #     fM_message = Wrench(force=Vector3(x=0.0, y=0.0, z=fM[0]), \
        #         torque=Vector3(x=fM[1], y=fM[2], z=fM[3]))
        # print(fM_message)
        # pub.publish(fM_message)

        rate.sleep()
    rover.file_time.close()
    rover.file_final_states.close()
    rover.file_states.close()
    # print('CONTROL: thread closed!')
    # print('Finished \n\n\n\n\n\n\n\n\n\n')
    
