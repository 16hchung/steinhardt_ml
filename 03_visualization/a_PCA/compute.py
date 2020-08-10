import numpy as np
from sklearn.decomposition import PCA
import pickle as pk

from util import dir_util, constants as C

def main():
  # Load data from example files.
  ps_test_names = dir_util.clean_features_paths02(pseudo=True)
  test_names    = dir_util.clean_features_paths02(istest=True)
  liq_names     = dir_util.clean_features_paths02(istest=True, liq=True)
  ps_X = np.loadtxt(ps_test_names.X.format('concat_'))[:,:]
  #ps_X = np.loadtxt(ps_test_names.unscaledX.format('concat_'))[:,:-28]
  #test_names = dir_util.clean_features_paths02(istest=True, lattice=C.lattices[0], temp=200)
  #X1    = np.loadtxt(test_names.X.format('concat_'))
  #test_names = dir_util.clean_features_paths02(istest=True, lattice=C.lattices[0], temp=1100)
  #X2    = np.loadtxt(test_names.X.format('concat_'))
  #X = np.row_stack((X1,X2))
  #y = np.loadtxt(dir_util.clean_features_paths02(istest=True, pseudo=False, liq=False).y.format('concat_'))
  #X = np.loadtxt(test_names.X.format('concat_'))
  #X = X[y==C.lattices[0].y_label]
  X    = np.loadtxt(test_names.X.format('concat_'))[:,:]
  #liqX = X
  liqX = np.loadtxt(liq_names.X.format('concat_'))[:,:]
  #ps_X = np.loadtxt(ps_test_names.X.format('concat_'))[:,-4:]
  #X    = np.loadtxt(test_names.X.format('concat_'))[:,-4:]
  #liqX = np.loadtxt(liq_names.X.format('concat_'))[:,-4:]

  # Compute PCA from  and save first two components.
  pca = PCA()
  # fit princip components from pseudo steinhardts, transform both pseudo and real to see overlay
  ps_X_pca = pca.fit(ps_X).transform(ps_X)
  X_pca = pca.transform(X)
  liqX_pca = pca.transform(liqX)
  ps_paths = dir_util.pca_data_paths03(pseudo=True)
  paths = dir_util.pca_data_paths03()
  liq_paths = dir_util.pca_data_paths03(liq=True)
  np.savetxt(paths.comp1, X_pca[:,0])
  np.savetxt(paths.comp2, X_pca[:,1])
  np.savetxt(ps_paths.comp1, ps_X_pca[:,0])
  np.savetxt(ps_paths.comp2, ps_X_pca[:,1])
  np.savetxt(liq_paths.comp1, liqX_pca[:,0])
  np.savetxt(liq_paths.comp2, liqX_pca[:,1])

  # Compute cummulative variance explained .
  var = pca.explained_variance_ratio_.cumsum()
  u = np.arange(1,len(var)+1)
  np.savetxt(paths.variance, np.transpose([u,var]), fmt='%2d %.5f', header='  n_components | explained variance (cummulative sum)')
  with open(paths.pca, 'wb') as f:
    pk.dump(pca, f)

if __name__=='__main__':
  main()
