import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pickle
import numpy as np
from utils.formal.gaussian_distribution import GaussianDistribution
from utils.observers import full_state_nonlinear_from_gaussian as fsn
from kinematic_lateral_model import Kinematic, l_f, l_r

data_filename = 'data_for_model.pkl'
with open(data_filename, 'rb') as f:
    data = pickle.load(f)

x_0 = data['x_0']
us = data['us']
x_gnd = data['x_gnd']

print(f'{x_0.miu=},\n{x_gnd=},{len(us)=}')

model = Kinematic()
dt = 1/20
W = GaussianDistribution(np.zeros(model.n), np.eye(model.n)*1e-6)
non_est = fsn.Estimator(model.ode, model.n, model.m, dt, W)
x_cur, sysd = non_est.estimate(x_0, us)

print(f'{x_cur.miu=}')
print(f'{l_f=}, {l_r=}')