import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from util import dir_util
from util import constants as cnst

# Load features and balance classes.
#X_fcc = loadtxt('../01_compute_features/data/X/X_fcc.dat')
#X_bcc = loadtxt('../01_compute_features/data/X/X_bcc.dat')
#X_hcp = loadtxt('../01_compute_features/data/X/X_hcp.dat')
#N_min = min([X_fcc.shape[0], X_bcc.shape[0], X_hcp.shape[0]])
#X_fcc = X_fcc[:N_min]
#X_bcc = X_bcc[:N_min]
#X_hcp = X_hcp[:N_min]
def load_and_balance():
  Xs = {}
  for latt in cnst.lattices:
    Xs[latt] = np.loadtxt(dir_util.all_features_path01(latt))
  N_min = min([x.shape[0] for x in Xs.values()])
  Xs = {l:x[:N_min] for l,x in Xs.items()}
  return Xs

# Create label vectors.
#y_fcc = zeros(X_fcc.shape[0])
#y_bcc = ones(X_bcc.shape[0])
#y_hcp = 2*ones(X_hcp.shape[0])
def make_labels(Xs):
  ys = {}
  for latt, X in Xs.items():
    ys[latt] = np.ones(X.shape[0]) * latt.y_label
  return ys

# Concatenate all data.
#X = row_stack((X_fcc,X_bcc,X_hcp))
#y = concatenate((y_fcc,y_bcc,y_hcp))
def combine_lattices_data(Xs, ys):
  X = np.row_stack(list(Xs.values()))
  y = np.concatenate(list(ys.values()))
  return X,y

# Split in test/train with shuffle and save unscaled features.
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=200000,shuffle=True)
#savetxt('data/X_unscaled_test.dat', X_test, fmt='%.10e')
#savetxt('data/X_unscaled_train.dat', X_train, fmt='%.10e')
#savetxt('data/y_test.dat', y_test, fmt='%d')
#savetxt('data/y_train.dat', y_train, fmt='%d')
#
## Scale features.
#scaler = StandardScaler().fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)
#joblib.dump(scaler,'data/scaler.pkl')
#
## Save data.
#savetxt('data/X_test.dat', X_test, fmt='%.10e')
#savetxt('data/X_train.dat', X_train, fmt='%.10e')
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
