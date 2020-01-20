import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import joblib

from util import dir_util
from util import constants as cnst

# Load features and balance classes.
def load_and_balance(pseudo=True):
  Xs = {}
  for latt in cnst.lattices:
    # default is to use pseudo data
    Xs[latt] = np.loadtxt(dir_util.all_features_path01(latt, pseudo=pseudo))
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
def split_and_save(ps_X, ps_y, X, y):
  # save unscaled features
  # NOTE: always going to train on pseudo steinhardts
  X_train, y_train = shuffle(ps_X, ps_y)
  # NOTE: split real steinhardts into validation (Eg for learning curves) and test sets
  X_val, X_test, y_val, y_test = train_test_split(X,y,test_size=.6,shuffle=True)
  train_names = dir_util.clean_features_paths02(pseudo=True)
  val_names   = dir_util.clean_features_paths02()
  test_names  = dir_util.clean_features_paths02(istest=True)
  np.savetxt(test_names.unscaledX , X_test, fmt='%.10e')
  np.savetxt(val_names.unscaledX  , X_val , fmt='%.10e')
  np.savetxt(train_names.unscaledX, X_train, fmt='%.10e')
  np.savetxt(test_names.y         , y_test, fmt='%d')
  np.savetxt(val_names.y          , y_val , fmt='%d')
  np.savetxt(train_names.y        , y_train, fmt='%d')

  # scale features and save
  scaler = StandardScaler().fit(X_train)
  X_train = scaler.transform(X_train)
  X_test  = scaler.transform(X_test)
  X_val   = scaler.transform(X_val)
  joblib.dump(scaler, dir_util.scaler_path02(pseudo=True))

  np.savetxt(test_names.X , X_test, fmt='%.10e')
  np.savetxt(val_names.X  , X_val , fmt='%.10e')
  np.savetxt(train_names.X, X_train,fmt='%.10e')

def main():
  ps_Xs = load_and_balance(pseudo=True)
  ps_ys = make_labels(ps_Xs)
  Xs = load_and_balance(pseudo=False)
  ys = make_labels(Xs)
  ps_X,ps_y = combine_lattices_data(ps_Xs, ps_ys)
  X,y = combine_lattices_data(Xs, ys)
  split_and_save(ps_X, ps_y, X,y)

if __name__=='__main__':
  main()
