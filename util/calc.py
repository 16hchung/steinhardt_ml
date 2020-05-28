from numpy import *
import numpy as np
import numpy.random as np_rnd
from numpy import linalg
import itertools
from scipy.linalg import norm
import scipy.stats
from scipy.special import sph_harm

from ovito.data import NearestNeighborFinder, CutoffNeighborFinder

from . import constants as cnst, dir_util

################## CARTESIAN COORDS ################################

def add_offsets(pipeline, data, scale=.06):
  # get first neighbor distance
  finder = NearestNeighborFinder(1, data)
  # arbitrarily use 10th atom as center
  first_neigh_d = list(finder.find(10))[0].distance
  
  # uniformly add noise to positions
  def pipeline_add_offsets(frame, data):
    positions = data.particles_.positions
    n_total_points = positions[:].shape[0]
    # generate unit vectors in random directions
    displc_vecs = np_rnd.randn(3, n_total_points).T
    #displc_vecs = np_rnd.uniform(low=-1.0, high=1.0, size=n_total_coords).reshape(positions.shape)
    norms = linalg.norm(displc_vecs, axis=1, keepdims=True)
    displc_vecs = displc_vecs / norms
    # generate uniformly distributed random displacement magnitudes to apply to displc_vecs
    mags = np_rnd.uniform(
        0, first_neigh_d * scale, 
        size=n_total_points
    ).reshape(norms.shape)
    displc_vecs = displc_vecs * mags
    # add displacement vectors to positions
    data.particles_.positions_ += displc_vecs

  pipeline.modifiers.append(pipeline_add_offsets)
  return pipeline.compute()
  
def relabel_wrong_neigh(y_valid, n_neigh):
  y_valid[y_valid < 0] = -1
  #for latt in cnst.lattices:
  #  if latt.n_neigh == n_neigh:
  #    continue
  #  y_valid[y_valid == latt.y_label] = -1
  return y_valid


################## STEINHARDT FUNCTIONS ###############################

# Compute Steinhardt order parameter for all atoms in data.
def compute_steinhardt(data,l,N_neigh,one_by_one=False):
  natoms = data.particles.count # Number of atoms.
  computed_steinhardt = zeros((natoms,len(l)))
  finder = NearestNeighborFinder(N_neigh,data) # Computes atom neighbor lists.
  # Loop over atoms to compute Steinhardt order paramter.
  for iatom in range(natoms):
    # Unroll neighbor distances.
    r_dim = 1 if one_by_one else N_neigh
    r_ij = zeros((r_dim, 3),dtype=float) # Distance to neighbors.
    ineigh = 0 # Neighbor counter.
    for i_neigh, neigh in enumerate(finder.find(iatom)):
      if one_by_one and i_neigh != N_neigh - 1:
        continue
      r_ij[ineigh][:] = neigh.delta
      ineigh += 1
    # Compute steinhardt for this atom
    computed_steinhardt[iatom][:] = steinhardt(r_ij,l)
  return computed_steinhardt

# Compute Steinhardt parameter for all orders in l.
def steinhardt(r,l):
  Q = zeros(len(l))
  for i in range(len(l)):
    q = zeros(2*l[i]+1,dtype=complex)
    for v in r:
      phi = arctan2(v[1],v[0])
      theta = arccos(v[2]/norm(v))
      q += sph_harm(arange(-l[i],l[i]+1),l[i],phi,theta)
    q /= r.shape[0]
    Q[i] = sqrt(real((4*pi/(2*l[i]+1)) * sum(q*conjugate(q))))
  return Q


################### RSF FUNCTIONS #####################################

def compute_rsf(data, sigma=.5):
  # Setup useful variables.
  natoms = data.particles.count # Number of atoms.
  N_rsf  = len(cnst.select_possible_n_neigh)
  max_n_neigh = max(cnst.select_possible_n_neigh)
  rsf = zeros((natoms, N_rsf))

  # Find mus based on avg distance of each n_neighbor in select_possible_n_neigh
  mus = zeros((natoms, N_rsf))
  near_finder = NearestNeighborFinder(max_n_neigh, data)
  for iatom in range(natoms):
    # get distances from each neighbor up till max n_neigh
    d_ij = zeros((max_n_neigh))
    for i_neigh, neigh in enumerate(near_finder.find(iatom)):
      d_ij[i_neigh] = neigh.distance
    # get avg distance for each n_neigh
    for i_n_neigh, n_neigh in enumerate(cnst.select_possible_n_neigh):
      mus[iatom,i_n_neigh] = mean(d_ij[:n_neigh])

  # r_cut should just be based on maximum mu
  r_cut = np.max(mus) + 2*sigma

  # Computes atoms' neighbor lists for rsf computation
  cut_finder = CutoffNeighborFinder(cutoff=r_cut, data_collection=data)
  
  # Loop over atoms to compute rsf
  for iatom in range(natoms):
    # Compute total number of neighbors of current atom.
    N_neigh = 0
    for neigh in cut_finder.find(iatom):
      N_neigh += 1
    # Unroll neighbor distances.
    d_ij = zeros(N_neigh,dtype=float) # Distance to neighbors.
    for i_neigh, neigh in enumerate(cut_finder.find(iatom)):
      d_ij[i_neigh] = neigh.distance
    # Compute RSF.
    for i in range(N_rsf):
      rsf[iatom,i] = sqrt(2*pi*sigma**2) * sum(scipy.stats.norm.pdf(d_ij,loc=mus[iatom,i],scale=sigma))

  return rsf
