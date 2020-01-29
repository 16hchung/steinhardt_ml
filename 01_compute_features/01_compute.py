import numpy as np
from ovito.io import import_file, export_file

from util import calc
from util import constants as cnst
from util import dir_util


def compute(latt, pseudo=.06, cart_only=False):
  steps = [0] if pseudo != None else np.arange(10000,20000+1000,1000) # Step range.
  l = np.arange(1,30+1)
  N_stein = len(l)
  # Iterate over steps.
  X = np.zeros((0,N_stein))
  for step in steps:
    pipeline = import_file(dir_util.dump_path_for_lattice00(latt, pseudo != None).format(step))
    data = pipeline.compute()
    if pseudo:
      data = calc.add_offsets(pipeline, data, position_noise_scaler=pseudo)
      export_file(
        data, dir_util.synth_carteasian_path01(latt), "lammps_dump",
        columns = ["Position.X", "Position.Y", "Position.Z"]
      )
      if cart_only:
        return
    X = np.vstack((X, calc.compute_steinhardt(data,l,latt.n_neigh)))
  np.savetxt(dir_util.all_features_path01(latt, pseudo != None), X, fmt='%.10e')

def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--latt', type=str, default=None)
  parser.add_argument('--pseudo_param', type=float, default=None)
  parser.add_argument('--cart_only', action='store_true', help='save cartesian then return before calculating features')
  args = parser.parse_args()
  
  lattices = cnst.lattices if args.latt == None else [cnst.str_to_latt[args.latt]]
  for latt in lattices:
    print(latt.name)
    compute(latt, args.pseudo_param, cart_only=args.cart_only)

if __name__=='__main__':
  main()
