import numpy as np
from tqdm import tqdm
from ovito.io import import_file, export_file

from util import calc
from util import constants as cnst
from util import dir_util

default_pseudo = .3

def compute_real_n_neigh(latt, l, N_stein, n_neigh):
  steps = np.arange(10000, 20000+1000, 1000)
  # Iterate over steps.
  X = np.zeros((0, N_stein))
  for ts in tqdm(steps):
    pipeline = import_file(dir_util.dump_path_for_lattice00(latt).format(ts))
    data = pipeline.compute()
    X = np.vstack((X, calc.compute_steinhardt(data, l, n_neigh)))
  np.savetxt(dir_util.all_features_path01(latt).format(n_neigh), X, fmt='%.10e')

def compute_real(latt, l, N_stein):
  for n_neigh in cnst.possible_n_neigh:
    compute_real_n_neigh(latt, l, N_stein, n_neigh)

def compute_synthetic(latt, l, N_stein, pseudo=default_pseudo):
  scales = np.linspace(.01, pseudo, num=10)
  X = np.zeros((0, N_stein))
  for s in tqdm(scales):
    pipeline = import_file(dir_util.dump_path_for_lattice00(latt, True).format(0))
    data = pipeline.compute()
    data = calc.add_offsets(pipeline, data, scale=s)
    #export_file(
    #  data, dir_util.synth_carteasian_path01(latt), "lammps_dump",
    #  columns = ["Position.X", "Position.Y", "Position.Z"]
    #)
    X = np.vstack((X, calc.compute_steinhardt(data, l, latt.n_neigh)))
  np.savetxt(dir_util.all_features_path01(latt, True).format(latt.n_neigh), X, fmt='%.10e')

def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--latt', type=str, default=None)
  parser.add_argument('--pseudo_param', type=float, default=None)
  parser.add_argument('--comp_both', action='store_true')
  args = parser.parse_args()
 
  # constants
  l = np.arange(1,30+1)
  N_stein = len(l)
  pseudo_param = args.pseudo_param if args.pseudo_param != None \
            else .3                if args.comp_both            \
            else None

  lattices = cnst.lattices if args.latt == None else [cnst.str_to_latt[args.latt]]
  for latt in lattices:
    print(latt.name)
    if pseudo_param != None or args.comp_both:
      print('computing synthetic')
      compute_synthetic(latt, l, N_stein, pseudo=pseudo_param)
    if pseudo_param == None or args.comp_both:
      print('computing real')
      compute_real(latt, l, N_stein)

if __name__=='__main__':
  main()
