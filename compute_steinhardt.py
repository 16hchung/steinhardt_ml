import os
from numpy import *
from scipy.stats import norm
from ovito.io import *
from ovito.data import *

from auxiliary import steinhardt

def compute_steinhardt(data,r_cut,l):
  # Setup useful variables.
  natoms = data.particles.count # Number of atoms.
  computed_steinhardt = zeros((natoms,len(l))) 

  # Computes atoms' neighbor lists.
  finder = CutoffNeighborFinder(cutoff=r_cut,data_collection=data)

  # Loop over atoms to compute rsf.
  for iatom in range(natoms):
    # Compute total number of neighbors of current atom.
    N_neigh = 0
    for neigh in finder.find(iatom):
      N_neigh += 1

    # Unroll neighbor distances.
    r_ij = zeros((N_neigh, 3),dtype=float) # Distance to neighbors.
    ineigh = 0 # Neighbor counter.
    for neigh in finder.find(iatom):
      r_ij[ineigh][:] = neigh.delta
      ineigh += 1 

    # Compute steinhardt for this atom
    computed_steinhardt[iatom][:] = steinhardt(r_ij, l)

  return computed_steinhardt

################################################################################
# Input parameters and setup.                                                  #
################################################################################

r_cut = 1.4 #no unit conversion okay?
l = arange(1,10+1)

################################################################################
# Compute steinhardt at each timestep.                                         #
################################################################################

structures = [
  ('bcc','lammps_scripts/02_crystals/data/dump_bcc_10000.dat'),
  ('fcc','lammps_scripts/02_crystals/data/dump_fcc_10000.dat'),
  ('hcp','lammps_scripts/02_crystals/data/dump_hcp_10000.dat'),
  ('liq','lammps_scripts/03_liquid/data/dump_liquid_10000.dat')
]

output_dir = 'data/from_sim'
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

for structure in structures:
  struct_type = structure[0]
  infile = structure[1]

  # Compute rsf.
  pipeline = import_file(infile)
  data = pipeline.compute()
  X = compute_steinhardt(data,r_cut,l)

  # Save features and labels.
  N_stein = len(X[0,:])
  savetxt('{}/{}_steinhardt.dat'.format(output_dir, struct_type),
          X, fmt='%.7e '*N_stein, header=" stein(%d)" % N_stein
  )

################################################################################
