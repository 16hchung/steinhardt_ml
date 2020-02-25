import numpy as np
from sklearn.preprocessing import StandardScaler

from util import dir_util
from util import constants as cnst

def get_scaler(wrt_latt, X_synth, y_synth):
  X_latt = X_synth[y_synth == wrt_latt.y_label]
  scaler = StandardScaler().fit(X_latt)
  np.savetxt(dir_util.zscore_data_path03(wrt_latt, synth=True), scaler.transform(X_latt))
  return scaler

def comp_zscores(wrt_latt, X_synth, y_synth, X_real, save_path):
  scaler = get_scaler(wrt_latt, X_synth, y_synth)
  X_real = scaler.transform(X_real)
  np.savetxt(save_path, X_real, fmt='%.10e')

def main():
  synth_paths = dir_util.clean_features_paths02(pseudo=True)
  X_synth = np.loadtxt(synth_paths.X.format('adapt_'))
  y_synth = np.loadtxt(synth_paths.y.format('adapt_'))
  real_paths = dir_util.clean_features_paths02(istest=True)
  for latt in cnst.lattices:
    save_path = dir_util.zscore_data_path03(latt)
    X_real = np.loadtxt(real_paths.X.format(latt.n_neigh))
    comp_zscores(latt, X_synth, y_synth, X_real, save_path)

if __name__=='__main__':
  main()
