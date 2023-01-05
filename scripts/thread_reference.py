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
    def __init__(self, rover):
        self.t = 0
        self.rover = rover
        self.finished = False
    
    def refresh_task(self):
        self.t = rover.t
        # print(self.t)
        if 0 < self.t < 12:
            rover.mode = 2
        elif 12 <= self.t < 24:
            rover.mode = 5
        if rover.alarm and not self.finished:
            self.finished = True
            rover.mode = 4


    

def thread_reference():
    # print('GUI: starting thread ..')
    ref = Reference(rover)
    rover.motor_on = True
    while rover.k_iter < rover.k_max and rover.on:
        ref.refresh_task()
        time.sleep(0.1)
    rover.on = False
    # print('GUI: thread closed!')