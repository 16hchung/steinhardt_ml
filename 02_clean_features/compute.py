import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from util import dir_util
from util import constants as cnst

# Load features and balance classes.
def load_and_balance():
  Xs = {}
  for latt in cnst.lattices:
    Xs[latt] = np.loadtxt(dir_util.all_features_path01(latt))
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
def split_and_save(X, y):
  # save unscaled features
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,shuffle=True)
  train_names = dir_util.clean_features_paths02()
  test_names  = dir_util.clean_features_paths02(istest=True)
  np.savetxt(test_names.unscaledX , X_test, fmt='%.10e')
  np.savetxt(train_names.unscaledX, X_train, fmt='%.10e')
  np.savetxt(test_names.y         , y_test, fmt='%d')
  np.savetxt(train_names.y        , y_train, fmt='%d')

  # scale features and save
  scaler = StandardScaler().fit(X_train)
  X_train = scaler.transform(X_train)
  X_test  = scaler.transform(X_test)
  joblib.dump(scaler, dir_util.scaler_path02())

  np.savetxt(test_names.X , X_test, fmt='%.10e')
  np.savetxt(train_names.X, X_train,fmt='%.10e')

def main():
  Xs = load_and_balance()
  ys = make_labels(Xs)
  X,y = combine_lattices_data(Xs, ys)
  split_and_save(X,y)

if __name__=='__main__':
  main()
