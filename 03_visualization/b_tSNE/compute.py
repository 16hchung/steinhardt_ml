import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from util import dir_util

################################################################################
# Input parameters and setup.                                                  #
################################################################################

#perplexity = int(sys.argv[1])
#
#M = 2000 # Number of data points used in tSNE.
#N_var_99 = 21 # Number of PCA components to explain 99% of the variance.
#
## Load data from example files and limit data points.
#X = loadtxt('../02_order_features/data/X_train.dat')
#y = loadtxt('../02_order_features/data/y_train.dat')
#X = X[:M]
#y = y[:M]
#
## Save labels used in tSNE.
#if perplexity == 10:
#  savetxt('data/y.dat', y)
perplexity_list = [10, 50, 200, 500, 1000]
default_perplexity = 10

def input_params_and_setup(perplexity, paths):
  M = 2000 # Number of data points used in tSNE.

  # Load data from example files and limit data points.
  fnames = dir_util.clean_features_paths02()
  X = np.loadtxt(fnames.X)
  y = np.loadtxt(fnames.y)
  X = X[:M]
  y = y[:M]

  # Save labels used in tSNE.
  if perplexity == default_perplexity:
    np.savetxt(paths.y, y, fmt='%i')
  return X, y

################################################################################
# Compute tSNE.                                                                #
################################################################################

#X_tsne = TSNE(perplexity=perplexity).fit_transform(X)
#savetxt('data/tSNE_%d.dat' % perplexity, X_tsne)
def compute_tsne(X, perplexity, paths, fname=None):
  if not fname:
    fname = paths.X_tmplt.format(perplexity)
  X_tsne = TSNE(perplexity=perplexity).fit_transform(X)
  np.savetxt(fname, X_tsne)

################################################################################
# Compute tSNE w/ PCA filter.                                                  #
################################################################################

#pca = PCA(n_components=N_var_99)
#X_pca = pca.fit(X).transform(X)
#
## Loop over perplexities.
#X_tsne = TSNE(perplexity=perplexity).fit_transform(X_pca)
#savetxt('data/tSNE_PCA_%d.dat' % perplexity, X_tsne)
def compute_tsne_with_PCA(X, perplexity, paths):
  N_var_99 = 21 # Number of PCA components to explain 99% of the variance.

  pca = PCA(n_components=N_var_99)
  X_pca = pca.fit(X).transform(X)
  compute_tsne(X_pca, perplexity, paths, paths.X_with_PCA_tmplt.format(perplexity))

################################################################################

def main(perplexity=default_perplexity):
  paths = dir_util.tSNE_data_paths03()
  
  X, y = input_params_and_setup(perplexity, paths)
  compute_tsne(X, perplexity, paths)
  compute_tsne_with_PCA(X, perplexity, paths)

def main_many():
  for p in tqdm(perplexity_list):
    main(p)

def has_run_already():
  from pathlib import Path
  data_tmplt_check = dir_util.tSNE_data_paths03().X_with_PCA_tmplt
  for p in perplexity_list:
    if not Path(data_tmplt_check.format(p)).exists():
      return False
  return True

if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-p', '--perplexity', type=int, default=default_perplexity)
  parser.add_argument('-m', '--many', action='store_true')
  args = parser.parse_args()

  if args.many:
    main_many()
  else:
    perplexity = args.perplexity
    main(perplexity)
