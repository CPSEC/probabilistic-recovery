from interval import interval, imath
DISTURBANCE_INTERVAL = interval[0, 0] # interval[-0.01, 0.01]

# Start helper functions
def convert_1D_list_to_intervals(list_1D):
    new_list_1D = []
    for element in list_1D:
        new_list_1D.append(interval([element, element]))
    return new_list_1D

def convert_2D_list_to_intervals(list_2D):
    new_list_2D = []
    for vector in list_2D:
        new_vector = []
        for element in vector:
            new_vector.append(interval([element, element]))
        new_list_2D.append(new_vector)
    return new_list_2D

def replace_safe_state_intervals_with_saved_value_intervals(x_interval, x_saved_interval, unsafeSensors_oneHot):
    for i, unsafe in enumerate(unsafeSensors_oneHot):
        if not unsafe:
            x_interval[i] = x_saved_interval[i]
    return x_interval

# Helper functions for operations on lists of intervals
def __mul__(list_of_intervals, constant):
    return [Interval*constant for Interval in list_of_intervals]

def __add__(list_of_intervals, another_list_of_intervals):
    assert len(list_of_intervals)==len(another_list_of_intervals), "You can't add 2 lists (of intervals) with different number of elements"
    new_list = []
    for i in range(len(list_of_intervals)):
        new_list.append(list_of_intervals[i]+another_list_of_intervals[i])
    return new_list

def __sum__(list_of_list_of_intervals):
    summed_list = list_of_list_of_intervals[0]
    for list_of_intervals in list_of_list_of_intervals[1:]:
        for i, interval in enumerate(list_of_intervals):
            summed_list[i] += interval
    return summed_list

def __add_to_every_element__(list_of_intervals, another_interval):
    return [Interval+another_interval for Interval in list_of_intervals]

# Helper functions for numerical integration
def RungeKutta_Integration(dynamics, x_interval, u_interval, step_size):
    x = x_interval
    u = u_interval
    t = None

    k_1 = dynamics(t, x, u)

    k_2 = dynamics(t, __add__(x, __mul__(k_1, step_size/2.0)) , u)

    k_3 = dynamics(t, __add__(x, __mul__(k_2, step_size/2.0)) , u)

    k_4 = dynamics(t, __add__(x, __mul__(k_3, step_size)) , u)

    x_next = __add__(x, __sum__( [__mul__(k_1, step_size/6.0), __mul__(k_2, 2.0*step_size/6.0), __mul__(k_3, 2.0*step_size/6.0), __mul__(k_4, step_size/6.0)] ))
    
    return __add_to_every_element__(x_next, DISTURBANCE_INTERVAL)

def Newton_Integration(dynamics, x_interval, u_interval, step_size):
    x = x_interval
    u = u_interval
    t = None
    x_dot = dynamics(t, x, u)
    x_next = __add__(x, __mul__(x_dot, step_size))
    
    return __add_to_every_element__(x_next, DISTURBANCE_INTERVAL)

# Helper functions for midpoint and intersection
def isIntersecting(Int1, Int2):
    if len(Int1 & Int2)==0:
        return False
    else:
        return True

def midpoint(Int):
    lower_bound = Int[0][0]
    upper_bound = Int[0][1]
    return (lower_bound+upper_bound)/2.0

# reachability
def reach(dynamics, x0_interval, Us_saved, Xs_saved, unsafeSensors_oneHot, step_size):
    if Xs_saved:
        assert len(Us_saved) == len(Xs_saved)-1, "There should be one less u vector than saved x vectors! ("
    
    total_steps = len(Us_saved)
    x_interval = x0_interval
    if Xs_saved:
        Xs_saved_interval = convert_2D_list_to_intervals(Xs_saved)
    Us_saved_interval = convert_2D_list_to_intervals(Us_saved)
    for i in range(total_steps):
        if  Xs_saved:
            x_interval = replace_safe_state_intervals_with_saved_value_intervals(x_interval, Xs_saved_interval[i], unsafeSensors_oneHot)
        u_interval = Us_saved_interval[i]
        
        # x_interval = Newton_Integration(dynamics, x_interval, u_interval, step_size)
        x_interval = RungeKutta_Integration(dynamics, x_interval, u_interval, step_size)
        #print(x_interval)
    return x_interval

if __name__ == '__main__':
    import sys
    sys.path.append('../../')      
    from simulators.nonlinear.continuous_stirred_tank_reactor import cstr_imath as cstr
    
    # This code won't run if this file is imported.
    # print(interval([1,1]) == interval([1])) # True
    x0_interval = [interval[1,1], interval[3,3]]
    #print(x0_interval)
    Us_saved = [ [0], [0.1], [1]]
    Xs_saved = [ [4,5], [6,7], [8,9], [10,11]]
    unsafeSensors_oneHot = [0,1]
    step_size = 0.1
    out = reach(cstr, x0_interval, Us_saved, Xs_saved, unsafeSensors_oneHot, step_size)
    
    print(f'{out=}')
    print(f'{midpoint(interval[5,9])=}')
