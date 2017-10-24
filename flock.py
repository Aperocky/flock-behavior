"""-----------------------------------------------------------------------------

    Creator: Rocky Li

    Purpose: Simulate Flocking behavior in robots.

    Detailed Explanation: The robots will stay in a group as they move towards
    targets. after acquiring a target, it will move towards another one.

-----------------------------------------------------------------------------"""

#-------------------------------------------------------------------------------

import sys, os
cdir = os.getcwd()
sys.path.append(cdir)
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
import matplotlib.animation as animation

#-------------------------------------------------------------------------------

"""-----------------------------------------------------------------------------
Calculate updated velocity based on an array of robot it (this robot) sees.
-----------------------------------------------------------------------------"""
def v_update(posarg, pos, target):

    # Remove itself from the list of the robots it saw
    for i in range(len(posarg)):
        if np.array_equal(pos, posarg[i]):
            posarg = np.delete(posarg, i, 0)
            break

    # Calculate relative position
    posarg[:, :2] -= pos[:2]

    # Distance of relative positions
    dist = np.sqrt(np.sum(posarg[:, :2] ** 2, axis=1))

    # Create normal vector pointing to relative position
    normvector = np.array([a/b for a,b in zip(posarg[:,:2],dist)])

    # Call the function involving distance.
    power = np.exp(-dist/3)*100 + (dist**2)/1000 - 2.5

    # Multiply the normal vector with function.
    poweredvector = np.array([a*b for a,b in zip(normvector, power)])

    # Sum up the vector.
    sumvector = np.sum(poweredvector, axis=0)

    # Implement target acquisition and move toward target
    unitv = target_aq(pos[:2], target[int(pos[4])])
    sumvector += (-norm(sumvector)/2-30)*unitv

    # Implement speed limit
    if norm(sumvector) > 2:
        sumvector = sumvector / norm(sumvector) * 2

    # Return the opposite of the vector we calculated.
    return -sumvector


"""-----------------------------------------------------------------------------
Given target, returns a normal vector pointing toward target from pos
-----------------------------------------------------------------------------"""
def target_aq(pos, target):
    rel = target - pos
    unitv = rel / norm(rel)
    return unitv


"""-----------------------------------------------------------------------------
Determine if robot have arrived on target
-----------------------------------------------------------------------------"""
def target_arrival(pos, target):
    dist = np.linalg.norm(pos[:2]-target[int(pos[4])])
    return dist < 15


"""-----------------------------------------------------------------------------
Class: Simulation
Purpose: Generate a simulation from starting positions and calls individual
calculation each time it is updated -- robots do not communicate with each other.
-----------------------------------------------------------------------------"""
class Simulation:

    def __init__(self,
        initial_state = [[1,1,1,1],[0,0,-1,1],[-1,1,0,1]],
        size = 1,
        target = [[200,0],[200,-200]]
        ):
        self.state = np.asarray(initial_state)
        self.size = size * np.ones(self.state.shape[0])
        self.target = np.asarray(target)

    # Process time
    def time_step(self, dt):

        D = squareform(pdist(self.state[:, :2]))
        blist = D<50

        for i in range(len(blist)):
            posarg = self.state[blist[i]]
            pos = self.state[i]
            self.state[i, 2:4] = v_update(posarg, pos, self.target)
            if target_arrival(self.state[i], self.target):
                if (int(self.state[i, 4]) < (len(self.target)-1)):
                    self.state[i, 4] += 1

        self.state[:, :2] += self.state[:, 2:4] * dt

#-------------------------------------------------------------------------------

#               END BASE CODE AND BEGIN THE ANIMATION MODULE

#-------------------------------------------------------------------------------

# Create a initial state that spreads
initial_state = np.random.rand(50,4) - 0.5
initial_state[:, :2] *= 100
initial_state = np.insert(initial_state, 4, 0, axis=1)
target = np.random.rand(10,2)
target *= 200
target[:, 1] = -target[:, 1]
target = np.vstack([[0,0],target])
# target = np.array([[0,0],[200,0],[200,-200]])
sim = Simulation(initial_state=initial_state, target=target)

# Create a figure
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
xlimits = 250
ylimits = 250
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-50, xlimits), ylim=(-ylimits, 50))
dots, = ax.plot([], [], 'bo', ms=5)
line, = ax.plot(target[:,0], target[:,1], '--')


"""-----------------------------------------------------------------------------
Function: Init
Purpose: Create the initial state of the animation.
-----------------------------------------------------------------------------"""
def init():

    global sim, line
    dots.set_data([], [])
    return dots, line

"""-----------------------------------------------------------------------------
Function: update
Purpose: Create each frame.
-----------------------------------------------------------------------------"""
def update(i):

    global sim, line
    sim.time_step(0.2)
    dots.set_data(sim.state[:,0], sim.state[:,1])
    return dots, line

ani = animation.FuncAnimation(fig, update, frames=600,
                              interval=20, init_func=init)

plt.show()
