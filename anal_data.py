import os, sys
cdir = os.getcwd()
sys.path.append(cdir)
import flock
import numpy as np


initial_state = np.random.rand(50,4) - 0.5
initial_state[:, :2] *= 100
initial_state = np.insert(initial_state, 4, 0, axis=1)
# target = np.random.rand(10,2)
# target *= 200
# target[:, 1] = -target[:, 1]
# target = np.vstack([[0,0],target])
target = np.array([[0,0],[200,0],[200,-200]])
sim = flock.Simulation(initial_state=initial_state, target=target)
