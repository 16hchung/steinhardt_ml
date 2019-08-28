import os
from numpy import *
from scipy.stats import norm
from ovito.io import *
from ovito.data import *
from sklearn.utils import shuffle

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
N_stein = len(l)

################################################################################
# Compute steinhardt at each timestep.                                         #
################################################################################

structures = [
  ('bcc','lammps_scripts/02_crystals/data/dump_bcc_{}.dat'),
  ('fcc','lammps_scripts/02_crystals/data/dump_fcc_{}.dat'),
  ('hcp','lammps_scripts/02_crystals/data/dump_hcp_{}.dat'),
  ('liq','lammps_scripts/03_liquid/data/dump_liquid_{}.dat')
]

ts_scale = 1000
train_ts_range = (10,20)
test_ts = 20

output_dir = 'data/from_sim'
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

def comp_stein_for_all_structures(all_data, ts):
    for structure in structures:
        struct_type = structure[0]
        infile = structure[1].format(ts)
      
        # Compute steinhardt
        pipeline = import_file(infile)
        data = pipeline.compute()
        X = compute_steinhardt(data,r_cut,l)

        all_data[struct_type] = vstack((all_data[struct_type], X))

def balance_classes(all_data):
    # Once we've computed steinhardt for all structures, find struct with min data points 
    # and truncate everything else to balance classes
    min_len = min(data.shape[0] for _,data in all_data.items())
    return {s:shuffle(data)[:min_len,:] for s,data in all_data.items()}

def save_to_files(all_data, suffix):
    for struct_type,X in all_data.items():
        # Save features and labels.
        savetxt('{}/{}_steinhardt_{}.dat'.format(output_dir, struct_type, suffix),
            X, fmt='%.7e '*N_stein, header=" stein(%d)" % N_stein
        )

# compute steinhardt for all train data (all timesteps except one)
all_data = {s:zeros((0,N_stein)) for [s,fname] in structures}
for ts in range(*train_ts_range):
  ts = ts_scale * ts
  comp_stein_for_all_structures(all_data, ts)
all_data = balance_classes(all_data)
save_to_files(all_data, 'train')

# compute steinhardt for validation set
all_data = {s:zeros((0,N_stein)) for [s,fname] in structures}
comp_stein_for_all_structures(all_data, test_ts * ts_scale)
save_to_files(all_data, 'val')

################################################################################
