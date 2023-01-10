#!/usr/bin/env python

from rover import rover, reset_uav

from gui import thread_gui
from thread_imu import thread_imu
from thread_gps import thread_gps
from thread_control import thread_control
from thread_log import thread_log
from thread_resend_control import thread_resend_control
from thread_reference import thread_reference

import numpy as np
import rospy
import std_msgs
import threading
import time
import sys


def run_uav(strategy, detection_delay, noise, isolation):

    rospy.init_node('uav', anonymous=True)
    reset_uav()

    # Create threads
    t1 = threading.Thread(target=thread_control, args=(detection_delay, strategy, noise, isolation,))
    t2 = threading.Thread(target=thread_imu)
    t3 = threading.Thread(target=thread_gps)
    t4 = threading.Thread(target=thread_reference)
    t5 = threading.Thread(target=thread_log)
    t6 = threading.Thread(target=thread_resend_control, args=(noise,))
    
    # Start threads.
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()

    # Wait until all threads close.
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    t6.join()


if __name__ == '__main__':
    strategy = int(sys.argv[1])
    isolation = int(sys.argv[2])
    detection_delay = float(sys.argv[3])
    noise = float(sys.argv[4])
    run_uav( strategy, detection_delay, noise, isolation )
