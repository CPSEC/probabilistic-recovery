import numpy as np

def compute_iae(time, states, t_detection, strip):
    integral = 0
    for i in range( 1, len(time)-1 ):
        if time[i] > t_detection:
            error = compute_distance( states[i], strip )
            integral = integral + error * ( time[i] - time[i-1] )
    return integral

def time_outside_strip(time, states, t_detection, strip):
    integral = 0
    for i in range( 1, len(time)-1 ):
        if time[i] > t_detection:
            outside = outside_strip( states[i], strip )
            integral = integral + outside * ( time[i] - time[i-1] )
    return integral


def time_to_arrive_strip(time, states, t_detection, strip):
    integral = 0
    for i in range( 1, len(time)-1 ):
        if time[i] > t_detection:
            outside = outside_strip( states[i], strip )
            integral = integral + outside * ( time[i] - time[i-1] )
            if not outside: break
    return integral

def arrived_strip(time, states, t_detection, t_reconfiguration, strip):
    reached = 0
    stayed = 0
    for i in range( 1, len(time) ):
        if time[i] >= t_detection and time[i] <= t_reconfiguration:
            outside = outside_strip( states[i], strip )
            if not outside: 
                reached = 1
        if time[i] == t_reconfiguration:
            if not outside_strip( states[i], strip ):
                stayed = 1
    return reached, stayed

def compute_distance(state, strip):
    strip_value = strip.l @ state
    distance = 0
    if not(strip.a < strip_value <  strip.b):
        distance1 = abs(strip_value - strip.a) 
        distance2 = abs(strip_value - strip.b)
        distance = min(distance1, distance2) / np.linalg.norm(strip.l, 2)
        # distance = 1
    return distance



def outside_strip(state, strip):
    strip_value = strip.l @ state
    outside = 0
    if not(strip.a < strip_value <  strip.b):
        outside = 1
    return outside