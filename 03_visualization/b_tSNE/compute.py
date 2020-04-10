import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from util import dir_util
from util import constants as cnst

##########################################################################
# Input parameters and setup.                                            #
##########################################################################

perplexity_list = [10, 50, 400, 1000]
default_perplexity = 10
M = 200*len(cnst.lattices)# Number of data points used in tSNE.

def input_params_and_setup(perplexity, paths, pseudo=True):
  # Load data from example files and limit data points.
  fnames = dir_util.clean_features_paths02(istest=not pseudo, pseudo=pseudo)
  X = np.loadtxt(fnames.X.format('concat_'))
  y = np.loadtxt(fnames.y.format('concat_'))
  X = X[:M]
  y = y[:M]

  # Save labels used in tSNE.
  if perplexity == default_perplexity:
    np.savetxt(paths.y, y, fmt='%i')
  return X, y

def split_to_real_pseudo(tSNE):
  return tSNE[:M], tSNE[M:]

##########################################################################
# Compute tSNE.                                                          #
##########################################################################

def compute_tsne(X, perplexity, paths, fname=None):
  if not fname:
    fname = paths.X_tmplt.format(perplexity)
  X_tsne = TSNE(perplexity=perplexity).fit_transform(X)
  np.savetxt(fname, X_tsne)

##########################################################################
# Compute tSNE w/ PCA filter.                                            #
##########################################################################

def compute_tsne_with_PCA(X, perplexity, paths):
  N_var_99 = 21 # Number of PCA components to explain 99% of the variance.

  pca = PCA(n_components=N_var_99)
  X_pca = pca.fit(X).transform(X)
  compute_tsne(X_pca, perplexity, paths, paths.X_with_PCA_tmplt.format(perplexity))

##########################################################################

def main(perplexity=default_perplexity):
  paths    = dir_util.tSNE_data_paths03()
  ps_paths = dir_util.tSNE_data_paths03(pseudo=True)
  
  X, y       = input_params_and_setup(perplexity, paths,    pseudo=False)
  ps_X, ps_y = input_params_and_setup(perplexity, ps_paths, pseudo=True)
  all_X = np.row_stack((X, ps_X))
  compute_tsne(all_X, perplexity, paths)
  #compute_tsne_with_PCA(all_X, perplexity, paths)

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
