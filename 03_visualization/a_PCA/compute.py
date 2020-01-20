import numpy as np
from sklearn.decomposition import PCA

from util import dir_util

def main():
  # Load data from example files.
  #X = loadtxt('../02_order_features/data/X_test.dat')
  #y = loadtxt('../02_order_features/data/y_test.dat')
  test_names = dir_util.clean_features_paths02(istest=True)
  X = np.loadtxt(test_names.X)

  # Compute PCA and save first two components.
  pca = PCA()
  X_pca = pca.fit(X).transform(X)
  paths = dir_util.pca_data_paths03()
  np.savetxt(paths.comp1, X_pca[:,0])
  np.savetxt(paths.comp2, X_pca[:,1])

  # Compute cummulative variance explained .
  var = pca.explained_variance_ratio_.cumsum()
  u = np.arange(1,len(var)+1)
  np.savetxt(paths.variance, np.transpose([u,var]), fmt='%2d %.5f', header='  n_components | explained variance (cummulative sum)')

if __name__=='__main__':
  main()
