import numpy as np
import numpy.random as np_rnd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import joblib
from tqdm import tqdm

from util import dir_util
from util import constants as cnst

n_test_pts = 10000
n_train_pts = 50000
N_keep = int(n_train_pts / len(cnst.lattices))

# Load features and balance classes.
def load_and_balance(pseudo, n_neigh=None, liq=False):
  Xs = {}
  for latt in cnst.lattices:
    # default is to use pseudo data
    neigh = latt.n_neigh if n_neigh == None else n_neigh
    Xs[latt] = np.loadtxt(dir_util.all_features_path01(latt, pseudo=pseudo, liq=liq).format(neigh))
  np_rnd.seed(0)
  #N_min = min([x.shape[0] for x in Xs.values()])
  Xs = {l:shuffle(x)[:N_keep] for l,x in Xs.items()}
  return Xs

# Create label vectors.
def make_labels(Xs):
  ys = {}
  for latt, X in Xs.items():
    ys[latt] = np.ones(X.shape[0]) * latt.y_label
  return ys

# Concatenate all data.
def combine_lattices_data(Xs, ys):
  X = np.row_stack(list(Xs.values()))
  y = np.concatenate(list(ys.values()))
  return X,y

def scale_data(X, n_neigh, fnames, scaler=None):
  # scale features and save
  if scaler == None:
    scaler = StandardScaler().fit(X)
    joblib.dump(scaler, dir_util.scaler_path02(pseudo=True).format(n_neigh))
  X = scaler.transform(X)
  return scaler, X

def process_n_neigh(fnames, pseudo, n_neigh=None, latt=None, temp=None, liq=False): # if default: generate pseudo/adaptive training 
  if latt==None or temp==None or pseudo:
    Xs = load_and_balance(pseudo, n_neigh, liq=liq)
    ys = make_labels(Xs)
    X, y = combine_lattices_data(Xs, ys)
  else:
    X = np.loadtxt(dir_util.all_features_path01(latt, pseudo=pseudo, temp=temp).format(n_neigh))
    X = X[:N_keep]
    y = np.ones(N_keep) * latt.y_label
  return X, y

def shuffle_all_and_save(Xs, ys, fnames, n_neighs, scaler=None, concat=False, save_y=True):
  shuff = shuffle(*Xs, *ys)
  Xs = shuff[:len(Xs)]
  ys = shuff[len(Xs):]

  def save_single(X, unscaledX, y, suffix):
      np.savetxt(fnames.unscaledX.format(suffix), unscaledX, fmt='%.10e')
      if save_y: np.savetxt(fnames.y.format(suffix),         y, fmt='%d')
      np.savetxt(fnames.X.format(suffix), X, fmt='%.10e')

  if concat:
    y = np.absolute(ys[0])
    unscaledX = np.concatenate(Xs, axis=1)
    X = unscaledX
    if scaler != None:
      #scaledXs = [scaler.transform(x) for x in Xs]
      #X = np.concatenate(scaledXs, axis=1)
      X = np.concatenate(Xs, axis=1)
      X = scaler.transform(X)
    save_single(X, unscaledX, y, 'concat_')
  else:
    for i, unscaledX in enumerate(Xs):
      y = ys[i]
      n_neigh = n_neighs[i]
      X = unscaledX if scaler == None else scaler.transform(unscaledX)
      save_single(X, unscaledX, y, n_neigh)
  return Xs, ys

def process_set(fnames, pseudo=False, scaler=None, scaler_path=None, concat=False, latt=None, temp=None, liq=False):
  Xs = []
  ys = []
  for neigh in cnst.possible_n_neigh:
    X, y = process_n_neigh(fnames, pseudo, neigh, latt, temp, liq=liq)

    incorrect_labels = [lbl for lbl, latt in cnst.lbl_to_latt.items() if latt.n_neigh != neigh]
    y[np.isin(y, incorrect_labels)] *= -1

    Xs.append(X)
    ys.append(y)
  if scaler == None and scaler_path != None:
    scaler, _ = scale_data(np.concatenate(Xs, axis=1), 'all_', fnames)
  Xs,ys = shuffle_all_and_save(Xs, ys, fnames, cnst.possible_n_neigh, scaler, concat, save_y=temp==None)
  return Xs, ys, scaler

def process_perf(scaler, latt):
  unscaled_fname = dir_util.perf_features_path(latt, scaled=False)
  scaled_fname   = dir_util.perf_features_path(latt, scaled=True)
  X = np.loadtxt(unscaled_fname).reshape(1, -1)
  #for i in range(16):
  #  start = i*cnst.n_features
  #  end = (i+1)*cnst.n_features
  #  X[0][start:end] = scaler.transform(X[0][start:end].reshape(1,-1))
  X = scaler.transform(X)
  np.savetxt(scaled_fname, X, fmt='%.10e')

def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--cat', action='store_true')
  parser.add_argument('--part', default='p1')
  args = parser.parse_args()

  if args.part == 'p1':
    # do synth
    print('processing synth training data')
    fnames = dir_util.clean_features_paths02(pseudo=True)
    scaler_path = dir_util.scaler_path02(pseudo=True)
    Xs, ys, scaler = process_set(fnames, pseudo=True, scaler_path=scaler_path, concat=args.cat)

    # do real looping thru possible n_neigh
    print('processing real test data')
    fnames = dir_util.clean_features_paths02(istest=True)
    process_set(fnames, pseudo=False, scaler=scaler, concat=args.cat)
  if args.part == 'p2':
    print('processing by latt and temp')
    scaler = joblib.load(dir_util.scaler_path02(pseudo=True).format('all_'))
    for latt in tqdm(cnst.lattices):
      for temp in range(latt.low_temp, latt.high_temp+latt.step_temp, latt.step_temp):
        fnames = dir_util.clean_features_paths02(istest=True, lattice=latt, temp=temp)
        process_set(fnames, pseudo=False, scaler=scaler, concat=args.cat, latt=latt, temp=temp)
  if args.part == 'liq':
    print('processing liq test data')
    scaler = joblib.load(dir_util.scaler_path02(pseudo=True).format('all_'))
    fnames = dir_util.clean_features_paths02(istest=True, liq=True)
    process_set(fnames, pseudo=False, scaler=scaler, concat=args.cat, liq=True)
  if args.part == 'perf':
    print('scaling perfect features')
    scaler = joblib.load(dir_util.scaler_path02(pseudo=True).format('all_'))
    for latt in tqdm(cnst.lattices):
      process_perf(scaler, latt)
    
if __name__=='__main__':
  main()
