import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import joblib

from util import dir_util
from util import constants as cnst

# Load features and balance classes.
def load_and_balance(n_neigh=None):
  Xs = {}
  for latt in cnst.lattices:
    # default is to use pseudo data
    neigh = latt.n_neigh if n_neigh == None else n_neigh
    Xs[latt] = np.loadtxt(dir_util.all_features_path01(latt, pseudo=n_neigh == None).format(neigh))
  N_min = min([x.shape[0] for x in Xs.values()])
  Xs = {l:x[:N_min] for l,x in Xs.items()}
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

# Split in test/train with shuffle and save unscaled features.
def split_and_save(X, y, fnames, n_neigh):
  # save unscaled features
  X, y= shuffle(X, y)
  # NOTE: split real steinhardts into validation (Eg for learning curves) and test sets
  np.savetxt(fnames.unscaledX.format(n_neigh), X, fmt='%.10e')
  np.savetxt(fnames.y.format(n_neigh),         y, fmt='%d')
  return X, y

def scale_data(X, n_neigh, fnames, scaler=None):
  # scale features and save
  if scaler == None:
    scaler = StandardScaler().fit(X)
    joblib.dump(scaler, dir_util.scaler_path02(pseudo=True).format(n_neigh))
  X = scaler.transform(X)

  np.savetxt(fnames.X.format(n_neigh), X, fmt='%.10e')
  return scaler

def process_n_neigh(fnames, n_neigh=None): # if default: generate pseudo/adaptive training 
  Xs = load_and_balance(n_neigh)
  ys = make_labels(Xs)
  X, y = combine_lattices_data(Xs, ys)
  X, y = split_and_save(X, y, fnames, 'adapt_' if n_neigh == None else n_neigh)
  return X, y

def main():
  # do synth
  fnames = dir_util.clean_features_paths02(pseudo=True)
  scaler_path = dir_util.scaler_path02(pseudo=True)
  X, _ = process_n_neigh(fnames)
  scaler = scale_data(X, 'adapt_', fnames)

  # do real looping thru possible n_neigh
  for neigh in cnst.possible_n_neigh:
    fnames = dir_util.clean_features_paths02(istest=True)
    X, _ = process_n_neigh(fnames, neigh)
    scale_data(X, neigh, fnames, scaler)

if __name__=='__main__':
  main()
