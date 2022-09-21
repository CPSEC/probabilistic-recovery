from rover import rover
from plot_utils import plot_data

import gi
import numpy as np
import os
import time
import rospy

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib, Gdk

class Reference():
    def __init__(self):
        self.t = 0
    
    def refresh_task(self):
        self.t = rover.t
        # print(self.t)
        if 0 < self.t < 5:
            rover.mode = 2
        elif 5 < self.t < 15:
            rover.mode = 5


    

def thread_reference():
    # print('GUI: starting thread ..')
    ref = Reference()
    rover.motor_on = True
    while not rospy.is_shutdown():
        ref.refresh_task()
        time.sleep(0.1)
    rover.on = False
    print('GUI: thread closed!')