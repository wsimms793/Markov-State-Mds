######################################################################################
#Importing relavent libraries.

import deeptime
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
import matplotlib
import scipy.stats
######################################################################################

######################################################################################
#Constructing the N x N array for the grid.

N = 20

position = np.zeros((N,N))

#Each element in the grid is numbered 0-224, starting at [0,0] and increasing along the cols and then rows.
#The mapping is bijective, so that [i,j] can be returned from this numbering. 
n = 0
for i in range(0,N):
    for j in range(0,N):
        position[i,j] = (i) + (j) + n*(N-1) 
    n += 1
######################################################################################
######################################################################################
#Defining the actual transition matrix.
def p(i, j):
    #Extracting rows and cols.
    row = i // N
    col = i % N

    #Probability of i -> i is zero as default.
    p_stay = 0

    #One step right
    if col < N-1:
        if j == i + 1:
            return 3/24
    else:
        p_stay += 3/24

    #One step left
    if col > 0:
        if j == i - 1:
            return 2/24
    else:
        p_stay += 2/24

    #Two steps right
    if col < N-2:
        if j == i + 2:
            return 2/24
    else:
        p_stay += 2/24

    #Two steps left
    if col > 1:
        if j == i - 2:
            return 2/24
    else:
        p_stay += 2/24

    #One step up
    if row < N-1:
        if j == i + N:
            return 5/24
    else:
        p_stay += 5/24

    #One step down
    if row > 0:
        if j == i - N:
            return 4/24
    else:
        p_stay += 4/24

    #Two steps up
    if row < N-2:
        if j == i + 2*N:
            return 5/24
    else:
        p_stay += 5/24

    #Two steps down
    if row > 1:
        if j == i - 2*N:
            return 1/24
    else:
        p_stay += 1/24

    #Probabilities from forbidden moves go to Probability i -> i.
    if j == i:
        return p_stay
    return 0


# Given a position, determine the next move using the t.p.f.
def move(X):

    #Uniform random number between 0-1.
    u = np.random.uniform(0, 1)

    #Assigning a move based on the t.p.f.
    cumulative_prob = 0

    for j in range(N*N):

        cumulative_prob += p(X, j)

        if u < cumulative_prob:
            return j

######################################################################################

######################################################################################
#Simulating the random walk.

#Random Seed.
np.random.seed(1234)

#Number of steps.
n = 125

#Array for positions.
X = np.zeros(n, dtype=int)


for i in range(0,n-1):     
    X[i+1] = move(X[i])
       
######################################################################################

######################################################################################
#Plotting the Random Walk.

#Extracting x-y cords.
x = X % N
y = X // N

#Plotting the trajectory and grid.
plt.plot(x, y, marker='o')
plt.xlim(-0.5, N)
plt.ylim(0.5, N)

plt.grid()
#Adding some useful points.
plt.plot(x[0], y[0], '*', markersize=12, label='Home')
plt.plot(N-1, N-1, '*', markersize=12, label='Bar')
plt.plot(x[n-1],y[n-1],'o',markersize = 12, label = 'endpoint')
plt.legend(loc='lower right')

######################################################################################
