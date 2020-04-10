import numpy as np
from sklearn.decomposition import PCA

from util import dir_util

def main():
  # Load data from example files.
  ps_test_names = dir_util.clean_features_paths02(pseudo=True)
  test_names    = dir_util.clean_features_paths02(istest=True)
  ps_X = np.loadtxt(ps_test_names.X.format('concat_'))
  X    = np.loadtxt(test_names.X.format('concat_'))

  # Compute PCA from  and save first two components.
  pca = PCA()
  # fit princip components from pseudo steinhardts, transform both pseudo and real to see overlay
  ps_X_pca = pca.fit(ps_X).transform(ps_X)
  X_pca = pca.transform(X)
  ps_paths = dir_util.pca_data_paths03(pseudo=True)
  paths = dir_util.pca_data_paths03()
  np.savetxt(paths.comp1, X_pca[:,0])
  np.savetxt(paths.comp2, X_pca[:,1])
  np.savetxt(ps_paths.comp1, ps_X_pca[:,0])
  np.savetxt(ps_paths.comp2, ps_X_pca[:,1])

  # Compute cummulative variance explained .
  var = pca.explained_variance_ratio_.cumsum()
  u = np.arange(1,len(var)+1)
  np.savetxt(ps_paths.variance, np.transpose([u,var]), fmt='%2d %.5f', header='  n_components | explained variance (cummulative sum)')

if __name__=='__main__':
  main()
