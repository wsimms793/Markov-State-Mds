%pip install pyemma
%pip install mdshare
#####################################
#Importanting relavent libraries.
import pyemma
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
matplotlib.rcParams.update({'font.size': 12})
import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt
import warnings
from mdshare import fetch
#####################################
#####################################
#Some compatability issue resolved.
import mdtraj
import types

if not hasattr(mdtraj, "version"):
    mdtraj.version = types.SimpleNamespace(version=mdtraj.__version__)
print(mdtraj.__version__)
print(mdtraj.version.version)
#####################################
#####################################
#Function for constructing a histogram from 2D data and plotting as a contour plot.
def plot_sampled_function(xall, yall, zall, ax=None, nbins=100, nlevels=20, cmap=plt.cm.bwr, cbar=True, cbar_label=None):
    # histogram data
    xmin = np.min(xall)
    xmax = np.max(xall)
    dx = (xmax - xmin) / float(nbins)
    ymin = np.min(yall)
    ymax = np.max(yall)
    dy = (ymax - ymin) / float(nbins)
    # bin data
    eps = x
    xbins = np.linspace(xmin - 0.5*dx, xmax + 0.5*dx, num=nbins)
    ybins = np.linspace(ymin - 0.5*dy, ymax + 0.5*dy, num=nbins)
    xI = np.digitize(xall, xbins)
    yI = np.digitize(yall, ybins)
    # result
    z = np.zeros((nbins, nbins))
    N = np.zeros((nbins, nbins))
    # average over bins
    for t in range(len(xall)):
        z[xI[t], yI[t]] += zall[t]
        N[xI[t], yI[t]] += 1.0

    with warnings.catch_warnings() as cm:
        warnings.simplefilter('ignore')
        z /= N
    # do a contour plot
    extent = [xmin, xmax, ymin, ymax]
    if ax is None:
        ax = plt.gca()
    ax.contourf(z.T, 100, extent=extent, cmap=cmap)
    if cbar:
        cbar = plt.colorbar()
        if cbar_label is not None:
            cbar.ax.set_ylabel(cbar_label)

    return ax
#####################################
######################################################################################
#Plotting the 2D P.D.F.
def plot_sampled_density(xall, yall, zall, ax=None, nbins=100, cmap=plt.cm.Blues, cbar=True, cbar_label=None):
    return plot_sampled_function(xall, yall, zall, ax=ax, nbins=nbins, cmap=cmap, cbar=cbar, cbar_label=cbar_label)
######################################################################################
######################################################################################
#Importing the Pentapeptide MD data.
topfile = fetch('pentapeptide-impl-solv.pdb')
traj_list = [fetch('pentapeptide-*-500ns-impl-solv.xtc')]
######################################################################################
######################################################################################
#Selecting features for further anaylsis.
# using the cos/sin transform
feat = coor.featurizer(topfile)
feat.add_backbone_torsions(cossin=True)
feat.add_sidechain_torsions(which=['chi1'], cossin=True)
#feat.add_distances(feat.pairs(feat.select_Heavy()))
# describe the features
# feat.describe()
######################################################################################
######################################################################################
#Determining the Dimension.
feat.dimension()
######################################################################################
######################################################################################
#Basic statistics.
inp = coor.source(traj_list, feat)
print('number of trajectories = ',inp.number_of_trajectories())
print('trajectory length = ',inp.trajectory_length(0))
print('trajectory time step = ', 500.0 / (inp.trajectory_length(0)-1),'ns')
print('number of dimension = ',inp.dimension())
######################################################################################
######################################################################################
#cumulative kinetic variance
tica_obj.cumvar
######################################################################################
######################################################################################
# here we do a little trick to ensure that eigenvectors always have the same sign structure.
# That's irrelevant to the analysis and just nicer plots - you can ignore it.
for i in range(2):
    if tica_obj.eigenvectors[0, i] > 0:
        tica_obj.eigenvectors[:, i] *= -1
######################################################################################
######################################################################################
#TICA output.
Y = tica_obj.get_output() # get tica coordinates
print('number of trajectories = ', np.shape(Y)[0])
print('number of frames = ', np.shape(Y)[1])
print('number of dimensions = ',np.shape(Y)[2])
######################################################################################
######################################################################################
#distribution on the two dominant TICA coordinates.
def plot_labels(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.text(-2, -4.7, '1', fontsize=20, color='black')
    ax.text(-1.2, -5, '2', fontsize=20, color='black')
    ax.text(-4.2, 1.5, '3', fontsize=20, color='black')
    ax.text(-0.1, 0, '4', fontsize=20, color='white')
mplt.plot_free_energy(np.vstack(Y)[:,0], np.vstack(Y)[:,1]);
######################################################################################
######################################################################################
#the trajectories looks like in the space of the first three TICA components.
matplotlib.rcParams.update({'font.size': 14})
dt = 0.1
plt.figure(figsize=(8,5))
ax1=plt.subplot(311)
x = dt*np.arange(Y[0].shape[0])
plt.plot(x, Y[0][:,0]); plt.ylabel('IC 1'); plt.xticks([]); plt.yticks(np.arange(-8, 4, 2))
ax1=plt.subplot(312)
plt.plot(x, Y[0][:,1]); plt.ylabel('IC 2'); plt.xticks([]);  plt.yticks(np.arange(-6, 4, 2))
ax1=plt.subplot(313)
plt.plot(x, Y[0][:,2]); plt.xlabel('time / ns'); plt.ylabel('IC 3'); plt.yticks(np.arange(-4, 6, 2));
######################################################################################
#######################
#K-means.
n_clusters = 250      # number of k-means clusters
clustering = coor.cluster_kmeans(Y,k=n_clusters)
dtrajs = clustering.dtrajs
mplt.plot_free_energy(np.vstack(Y)[:,0], np.vstack(Y)[:,1])
cc_x = clustering.clustercenters[:,0]
cc_y = clustering.clustercenters[:,1]
plt.plot(cc_x,cc_y, linewidth=0, marker='o', markersize=5, color='black')
#######################
######################################
#Implied timescales.
its = msm.timescales_msm(dtrajs, lags=200, nits=10)
matplotlib.rcParams.update({'font.size': 14})
mplt.plot_implied_timescales(its, ylog=False, units='steps', linewidth=2)
plt.xlim(0, 40); plt.ylim(0, 120);
#######################################
#Error Bars.
its = msm.timescales_msm(dtrajs, lags=100, nits=10, errors='bayes', n_jobs=-1)
plt.figure(figsize=(8,5))
matplotlib.rcParams.update({'font.size': 14})
mplt.plot_implied_timescales(its, show_mean=False, ylog=False, dt=0.1, units='ns', linewidth=2)
plt.xlim(0, 5); plt.ylim(0.1,22);
#######################################
#######################################
#Esimating the MSM.
msm_lag = 12
M = msm.estimate_markov_model(dtrajs, msm_lag)
print('fraction of states used = ', M.active_state_fraction)
print('fraction of counts used = ', M.active_count_fraction)
# test MSM
M = msm.bayesian_markov_model(dtrajs, msm_lag)
ck = M.cktest(4, mlags=11, err_est=False)
matplotlib.rcParams.update({'font.size': 14})
mplt.plot_cktest(ck, diag=True, figsize=(7,7), layout=(2,2), padding_top=0.1, y01=False, padding_between=0.3, dt=0.1, units='ns')
#######################################
#######################################
# ... therefore we take the statistical weight of each simulation timestep (also available from the MSM object)
# and use that to create a contour plot
xall = np.vstack(Y)[:,0]
yall = np.vstack(Y)[:,1]
W = np.concatenate(M.trajectory_weights())
# TODO: make this an MSM-weighted free energy plot
matplotlib.rcParams.update({'font.size': 12})
mplt.plot_free_energy(xall, yall)
plot_labels()
#######################################
#######################################
# analyze the slowest processes by looking at the distribution of states along the first 3 eigenvectors.
# project eigenvectors
proj_ev_all = [np.hstack([M.eigenvectors_right()[:,i][dtraj] for dtraj in M.discrete_trajectories_full])
               for i in range(1, 10)]
fig, axes = plt.subplots(1, 3, figsize=(16,4))
for i, ax in enumerate(axes):
    plot_sampled_function(xall, yall, proj_ev_all[i], ax=ax, cbar=False, cmap=plt.cm.Blues)
    plot_labels(ax)
#######################################
#######################################
#Further Coarse Graining.
n_sets = 4
M.pcca(n_sets)
pcca_dist = M.metastable_distributions
membership = M.metastable_memberships  # get PCCA memberships
# memberships over trajectory
dist_all = [np.hstack([pcca_dist[i,:][dtraj] for dtraj in M.discrete_trajectories_full]) for i in range(n_sets)]
mem_all = [np.hstack([membership[:,i][dtraj] for dtraj in M.discrete_trajectories_full]) for i in range(n_sets)]
fig, axes = plt.subplots(1, 4, figsize=(16, 3))
matplotlib.rcParams.update({'font.size': 12})
axes = axes.flatten()

np.seterr(invalid='warn')
for k in range(n_sets):
        plot_sampled_density(xall, yall, dist_all[k], ax=axes[k], cmap=plt.cm.Blues, cbar=False)
pcca_samples = M.sample_by_distributions(pcca_dist, 10)
coor.save_trajs(inp, pcca_samples, outfiles=['./data/pcca1_10samples.xtc','./data/pcca2_10samples.xtc',
                                             './data/pcca3_10samples.xtc','./data/pcca4_10samples.xtc'])
#######################################
##############################################
#Coarse-grained kinetic Model using a Hidden Markov Model
hmm = M.coarse_grain(4)
hmm.stationary_distribution
hmm.transition_matrix
pos=np.array([[-3, -4], [0, -5], [-4.2, 0.5], [0, 0]])
mplt.plot_markov_model(hmm, pos=pos, minflux=4e-4, arrow_label_format='%.3f')
plt.gca().set_frame_on(False)
##############################################




