from numpy import *
import numpy.random as np_rnd
from numpy import linalg
import itertools
from scipy.linalg import norm
from scipy.special import sph_harm

from ovito.data import NearestNeighborFinder

from . import constants

# Compute Steinhardt order parameter for all atoms in data.
def compute_steinhardt(data,l,N_neigh):
  natoms = data.particles.count # Number of atoms.
  computed_steinhardt = zeros((natoms,len(l)))
  finder = NearestNeighborFinder(N_neigh,data) # Computes atom neighbor lists.
  # Loop over atoms to compute Steinhardt order paramter.
  for iatom in range(natoms):
    # Unroll neighbor distances.
    r_ij = zeros((N_neigh, 3),dtype=float) # Distance to neighbors.
    ineigh = 0 # Neighbor counter.
    for neigh in finder.find(iatom):
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

def add_offsets(pipeline, data):
  position_noise_scaler = .06
  # get first neighbor distance
  finder = NearestNeighborFinder(1, data)
  # arbitrarily use 10th atom as center
  first_neigh_d = list(finder.find(10))[0].distance
  
  # uniformly add noise to positions
  def add_single_offset(frame, data):
    positions = data.particles_.positions
    n_total_coords = positions[:].size 
    # generate unit vectors in random directions
    displc_vecs = np_rnd.uniform(size=n_total_coords).reshape(positions.shape)
    norms = linalg.norm(displc_vecs, axis=1, keepdims=True)
    displc_vecs = displc_vecs / norms
    # generate uniformly distributed random displacement magnitudes to apply to displc_vecs
    mags = np_rnd.uniform(
        0, first_neigh_d * position_noise_scaler, 
        size=positions.shape[0]
    ).reshape(norms.shape)
    displc_vecs = displc_vecs * mags
    # add displacement vectors to positions
    data.particles_.positions_ += displc_vecs

  pipeline.modifiers.append(add_single_offset)
  return pipeline.compute()
  

