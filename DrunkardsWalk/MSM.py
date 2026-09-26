%pip install deeptime
######################################################################################
#Importing relavent libraries.
import deeptime
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
import matplotlib
import scipy.stats
import networkx as nx
import deeptime.markov as markov
######################################################################################
######################################################################################
#Loading our Random walk trajectory.
X = np.load("X.npy")
######################################################################################
######################################################################################
trajectory = X
#Estimating the Transition matrix.
estimator = markov.TransitionCountEstimator(
    lagtime=1,
    count_mode="sliding"
)

#Creating the count-matrix.
counts = estimator.fit(trajectory).fetch_model()  
Count_matrix = counts.count_matrix


#Estimating the T.P matrix from the count matrix.
estimator = markov.msm.MaximumLikelihoodMSM(
    reversible=False,
    stationary_distribution_constraint=None, lagtime = 1
)

msm = estimator.fit(counts).fetch_model()
#Extracting the transition Matrix.
P_est = msm.transition_matrix


######################################################################################
######################################################################################
# Given a position, determine the next move using the estimated t.p.f.
#The grid length = grid width = N. 

#This extracts every state which has an element in the estimated t.p. matrix.
symbols = msm.state_symbols()
N = 20

#Since P_est only contains active states we rebuild the full estimated t.p. matrix.

P_est_full = np.zeros((N*N, N*N))

for i in range(len(symbols)):
    for j in range(len(symbols)):
        P_est_full[symbols[i], symbols[j]] = P_est[i, j]

def move(X):
    #Uniform random number between 0-1.
    u = np.random.uniform(0, 1)

    #Assigning a move based on the t.p.f.
    cumulative_prob = 0

    for j in range(N*N):
        cumulative_prob += P_est_full[X, j]

        if u < cumulative_prob:
            return j

######################################################################################
#Array for positions.

#Number of steps
n = 100000

Y = np.zeros(n, dtype=int)

for i in range(0,n-1):     
    Y[i+1] = move(Y[i])
######################################################################################
#Estimating the Invariant distrubution for Bottom left,right and Top left, right.
BL = 0
BR = 0
TL = 0
TR = 0
#Midpoint.
mid = N // 2
#Summing probabilities in each grid corner.
for i, y in enumerate(Y):

    row = y // N
    col = y % N

    if row < mid and col < mid:
        BL += 1

    elif row < mid and col >= mid:
        BR += 1

    elif row >= mid and col < mid:
        TL += 1

    else:
        TR += 1

#Collecting probabilites.
Estimated_Probs = [BL / n,BR / n, TL/n, TR/n]
######################################################################################
#Plotting this distrubition acorss Macro states.

fig, ax = plt.subplots()

Macrostates = ['Bottom Left', 'Bottom Right', 'Top Left', 'Top Right']
bar_labels = ['Bottom Left', 'Bottom Right', 'Top Left', 'Top Right']
bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

ax.bar(Macrostates, Estimated_Probs, label=bar_labels, color=bar_colors)

ax.set_ylabel('Estimated Invariant distrubtion')

ax.legend(title='Macrostate')




######################################################################################
