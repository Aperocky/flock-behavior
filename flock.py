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
    power = np.exp(-dist/4)*100 + (dist**2)/1000 - 2.5

    # Multiply the normal vector with function.
    poweredvector = np.array([a*b for a,b in zip(normvector, power)])

    # Sum up the vector.
    sumvector = np.sum(poweredvector, axis=0)

    # Implement target acquisition and move toward target
    if pos[5] == 0:
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
        self.size = size
        self.sizes = size * np.ones(self.state.shape[0])
        self.target = np.asarray(target)

    # Process time
    def time_step(self, dt):

        # Calculate collision numbers
        D = squareform(pdist(self.state[:, :2]))
        blist = D<50
        cnum = (np.sum(D<1)-len(self.state))/2
        # mask = np.ones(D.shape, dtype=bool)
        # np.fill_diagonal(mask, 0)
        # mindist = np.amin(D[mask])

        # Calculate Central Point
        centpoint = np.sum(self.state[:,:2], axis=0)/len(self.state)

        # Calculate distance average
        rel_dist = self.state[:,:2] - centpoint
        abs_dist = np.sqrt(np.sum(rel_dist**2, axis=1))
        avg_dist = np.sum(abs_dist)/len(abs_dist)

        for i in range(len(blist)):
            posarg = self.state[blist[i]]
            pos = self.state[i]
            self.state[i, 2:4] = v_update(posarg, pos, self.target)
            if target_arrival(self.state[i], self.target):
                if (int(self.state[i, 4]) < (len(self.target)-1)):
                    self.state[i, 4] += 1

        self.state[:, :2] += self.state[:, 2:4] * dt
        return cnum, avg_dist

#-------------------------------------------------------------------------------

#               END BASE CODE AND BEGIN THE ANIMATION MODULE

#-------------------------------------------------------------------------------

if __name__=="__main__":
    savefile = ''
    if len(sys.argv) > 2:
        savefile = sys.argv[2]
    if len(sys.argv) > 1:
        numer = int(sys.argv[1])

    # Create a initial state that spreads
    initial_state = np.random.rand(numer,4) - 0.5
    initial_state[:, :2] *= 100
    initial_state = np.insert(initial_state, 4, 0, axis=1)
    initial_state = np.insert(initial_state, 5, 0, axis=1)

    """ This determines if guided by only a few """
    numguide = int(numer/4)
    initial_state[numguide:, 5] = 1

    """ Random targets fun """
    # target = np.random.rand(10,2)
    # target *= 200
    # target[:, 1] = -target[:, 1]
    # target = np.vstack([[0,0],target])

    target = np.array([[0,0],[200,0],[200,-200]])
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
    text = ax.text(10, -100, '')
    text2 = ax.text(10, -120, '')
    text3 = ax.text(10, -140, '')
    ctotal = 0
    steps = 0

    # Get Average Distance
    avgdists = []

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

        global sim, line, ctotal, avgdists, steps
        cnum, avgdist = sim.time_step(0.15)
        avgdists.append(avgdist)
        # print(fig.dpi)
        # print(np.array(sim.size))
        # print(fig.get_figwidth())
        # print(np.diff(ax.get_xbound()))
        # print(int(np.array(fig.dpi * 2 * sim.size * fig.get_figwidth()/np.diff(ax.get_xbound())[0])))
        ms = int(fig.dpi * sim.size * fig.get_figwidth()/np.diff(ax.get_xbound())[0])
        ctotal += cnum
        steps += 1
        dots.set_data(sim.state[:,0], sim.state[:,1])
        text2.set_text("Step %d" % steps)
        text.set_text("Total number of collisions: %d" % ctotal)
        dots.set_markersize(ms)
        return dots, line

    ani = animation.FuncAnimation(fig, update, frames=2000,
                                  interval=20, init_func=init)
    if not savefile == '':
        ani.save(savefile+".mp4", writer='ffmpeg', dpi=120)
    else:
        plt.show()

    """-----------------------------------------------------------------------------
    Show the average distance to center.
    -----------------------------------------------------------------------------"""
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    xtick = np.arange(len(avgdists))*0.15
    ax.plot(xtick, avgdists)
    ax.set_ylabel("Average distance to center (m)")
    ax.set_xlabel("Seconds")
    ax.set_ylim(0, 50)
    ax.grid(True)
    if not savefile == '':
        fig1.savefig(savefile+".png")
    else:
        plt.show()
