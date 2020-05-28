import numpy as np
import numpy.random as np_rnd
from tqdm import tqdm
from ovito.io import import_file, export_file
from sklearn.utils import shuffle

from util import calc
from util import constants as cnst
from util import dir_util

default_pseudo = .3
one_by_one = False

def compute_real_n_neigh(latt, l, N_stein, n_neigh, liq=False, rsf=False):
  steps = np.arange(10000, 100000+10000, 10000)

  def compute_one_temp(temp):
    X = np.zeros((0, N_stein))
    for ts in tqdm(steps):
      pipeline = import_file(dir_util.dump_path_for_lattice00(latt, temp=temp, liq=liq).format(ts))
      data = pipeline.compute()
      if rsf:
        X = np.vstack((X, calc.compute_rsf(data)))
      else:
        X = np.vstack((X, calc.compute_steinhardt(data, l, n_neigh, one_by_one=one_by_one)))
    dest = dir_util.all_features_path01(latt, temp=temp, liq=liq, rsf=rsf).format('sel' if rsf else n_neigh)
    np.savetxt(dest, X, fmt='%.10e')
    return X

  # Iterate over steps.
  Xs = []
  if liq:
    Xs = [compute_one_temp(None)]
  else:
    for temp in tqdm(range(latt.low_temp, latt.high_temp + latt.step_temp, latt.step_temp)):
      X = compute_one_temp(temp)
      Xs.append(X)
  X = shuffle(np.vstack(Xs))
  dest = dir_util.all_features_path01(latt, liq=liq, rsf=rsf).format('sel' if rsf else n_neigh)
  np.savetxt(dest, X, fmt='%.10e')

def compute_real(latt, l, N_stein, liq=False, rsf=False):
  if rsf:
    compute_real_n_neigh(latt, l, N_stein, None, liq, rsf=True)
    return

  for n_neigh in cnst.possible_n_neigh:
    compute_real_n_neigh(latt, l, N_stein, n_neigh, liq)

def compute_perfect_n_neigh(latt, l, N_stein, n_neigh, rsf=False):
    pipeline = import_file(dir_util.dump_path_for_lattice00(latt, True).format(0))
    data = pipeline.compute()
    if rsf:
      X = calc.compute_rsf(data)
    else:
      X = calc.compute_steinhardt(data, l, n_neigh)
    np.savetxt(
        dir_util.all_features_path01(latt, pseudo=True, perfect=True, rsf=rsf).format('sel' if rsf else n_neigh),
        X, fmt='%.10e'
    )

def compute_perfect(latt, l, N_stein, rsf=False):
  if rsf:
    compute_perfect_n_neigh(latt, l, N_stein, None, rsf=True)
    return
 
  for n_neigh in tqdm(cnst.possible_n_neigh):
    compute_synthetic_n_neigh(latt, l, N_stein, n_neigh)

def compute_synthetic_n_neigh(latt, l, N_stein, pseudo, n_neigh, rsf=False):
  scales = np.linspace(.01, pseudo, num=15)
  X = np.zeros((0, N_stein))
  np_rnd.seed(0)
  for s in tqdm(scales):
    pipeline = import_file(dir_util.dump_path_for_lattice00(latt, True).format(0))
    data = pipeline.compute()
    data = calc.add_offsets(pipeline, data, scale=s)
    #export_file(
    #  data, dir_util.synth_carteasian_path01(latt), "lammps_dump",
    #  columns = ["Position.X", "Position.Y", "Position.Z"]
    #)
    if rsf:
      X = np.vstack((X, calc.compute_rsf(data)))
    else:
      X = np.vstack((X, calc.compute_steinhardt(data, l, n_neigh, one_by_one=one_by_one)))
  print(np_rnd.randn(1))
  X = shuffle(X)
  np.savetxt(
      dir_util.all_features_path01(latt, pseudo=True, rsf=rsf).format('sel' if rsf else n_neigh), 
      X, fmt='%.10e'
  )

def compute_synthetic(latt, l, N_stein, pseudo=default_pseudo, rsf=False):
  if rsf:
    compute_synthetic_n_neigh(latt, l, N_stein, pseudo, None, rsf=True)
    return

  for n_neigh in tqdm(cnst.possible_n_neigh):
    compute_synthetic_n_neigh(latt, l, N_stein, pseudo, n_neigh)

def main():
  global one_by_one
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--latt', type=str, default=None) # in effect required unless --perfect
  parser.add_argument('--pseudo_param', type=float, default=None)
  parser.add_argument('--comp_both', action='store_true') # DEPRECATED
  parser.add_argument('--one_by_one', action='store_true') # DEPRECATED
  parser.add_argument('--liq', action='store_true')
  parser.add_argument('--perfect', action='store_true')
  parser.add_argument('--rsf', action='store_true') # can be True or False with any combo of other args
  args = parser.parse_args()
  '''
  NOTE: Need to clean up, but for now, note that....
  For each latt, must run: (6*3*2 = 36 runs)
    --latt <latt> --pseudo_param .35 <with and without --rsf>
    --latt <latt> --liq <with and without --rsf>
    --latt <latt> <with and without --rsf>
  Also run: (2 run)
  --perfect <with and without --rsf>   # note: don't need to specify lattice bc runs short enough
  '''
 
  # constants
  one_by_one = args.one_by_one
  l = np.arange(1,cnst.n_features+1)
  N_stein = len(cnst.select_possible_n_neigh) if args.rsf else len(l)
  pseudo_param = args.pseudo_param if args.pseudo_param != None \
            else .3                if args.comp_both            \
            else None

  lattices = cnst.lattices if args.latt == None else [cnst.str_to_latt[args.latt]]
  for latt in lattices:
    print(latt.name)
    if args.perfect:
      print('computing perfect')
      compute_perfect(latt, l, N_stein, rsf=args.rsf)
      continue
    if args.liq:
      print('computing real liquid')
      compute_real(latt, l, N_stein, liq=True, rsf=args.rsf)
      continue
    if pseudo_param != None or args.comp_both:
      print('computing synthetic')
      compute_synthetic(latt, l, N_stein, pseudo=pseudo_param, rsf=args.rsf)
    if pseudo_param == None or args.comp_both:
      print('computing real')
      compute_real(latt, l, N_stein, rsf=args.rsf)

if __name__=='__main__':
  main()
