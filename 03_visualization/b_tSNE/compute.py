import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

from util import dir_util
from util import constants as cnst

##########################################################################
# Input parameters and setup.                                            #
##########################################################################

perplexity_list = [10, 50, 400, 1000]
default_perplexity = 10
M = 100*len(cnst.lattices)# Number of data points used in tSNE.

def input_params_and_setup(perplexity, paths, pseudo=True):
  # Load data from example files and limit data points.
  fnames = dir_util.clean_features_paths02(istest=not pseudo, pseudo=pseudo)
  X = np.loadtxt(fnames.X.format('concat_'))
  y = np.loadtxt(fnames.y.format('concat_'))
  X,y = shuffle(X,y)
  X = X[:M]
  y = y[:M]

  # Save labels used in tSNE.
  if perplexity == default_perplexity:
    np.savetxt(paths.y, y, fmt='%i')
  return X, y

def hcp_input_params_and_setup(perplexity, paths, pseudo=True):
  # get hcp for all temps
  Xs = []
  hcp = cnst.str_to_latt['hcp']
  for T in range(hcp.low_temp, hcp.high_temp+hcp.step_temp, hcp.step_temp):
    fname = dir_util.clean_features_paths02(
        istest=not pseudo, pseudo=pseudo, lattice=hcp, temp=T
    )
    Xs.append(np.loadtxt(fname.X.format('concat_')))
  X = np.row_stack(Xs)
  y = np.full((100), hcp.y_label)
  X = shuffle(X)[:100]
  
  # get liquids
  fnames = dir_util.clean_features_paths02(istest=True, pseudo=False, liq=True)
  Xliq= np.loadtxt(fnames.X.format('concat_'))
  yliq= np.loadtxt(fnames.y.format('concat_'))
  Xliq,yliq = shuffle(Xliq,yliq)
  Xliq = Xliq[:M]
  yliq = yliq[:M] * -1

  all_y = np.concatenate((y, yliq))
  all_X = np.row_stack((X, Xliq))
  if perplexity == default_perplexity:
    np.savetxt(paths.y, all_y, fmt='%i')

  return all_X, all_y

def split_to_real_pseudo(tSNE):
  return tSNE[:M], tSNE[M:]

##########################################################################
# Compute tSNE.                                                          #
##########################################################################

def compute_tsne(X, perplexity, paths, fname=None):
  if not fname:
    fname = paths.X_tmplt.format(perplexity)
  X_tsne = TSNE(perplexity=perplexity, n_iter=10000, verbose=2, n_jobs=-1).fit_transform(X)
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

def main(perplexity=default_perplexity, hcp_liq=False):
  paths    = dir_util.tSNE_data_paths03()
  ps_paths = dir_util.tSNE_data_paths03(pseudo=True)
 
  if hcp_liq:
    all_X, _ = hcp_input_params_and_setup(perplexity, paths, pseudo=False)
  else:
    X, y       = input_params_and_setup(perplexity, paths,    pseudo=False)
    ps_X, ps_y = input_params_and_setup(perplexity, ps_paths, pseudo=True)
    #TODO make sure that ps_y is all negative
    all_X = np.row_stack((X, ps_X))
  compute_tsne(all_X, perplexity, paths)
  #compute_tsne_with_PCA(all_X, perplexity, paths)

def main_many(hcp_liq):
  for p in tqdm(perplexity_list):
    main(p, hcp_liq)

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
  parser.add_argument('--hcp_liq', action='store_true')
  args = parser.parse_args()

  if args.many:
    main_many(args.hcp_liq)
  else:
    perplexity = args.perplexity
    main(perplexity, args.hcp_liq)
