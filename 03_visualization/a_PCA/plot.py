import matplotlib.pyplot as plt                 
import numpy as np

from util import dir_util
from util import constants as cnst

################################################################################
# Load and process data.                                                       #
################################################################################

def load_and_process_data(pseudo=True):
  M = 20000 # Number of points to be plotted.

  # Load results.
  data_names = dir_util.pca_data_paths03(pseudo=pseudo)
  PCA_1 = np.loadtxt(data_names.comp1)
  PCA_2 = np.loadtxt(data_names.comp2)
  y = np.loadtxt(dir_util.clean_features_paths02(istest=not pseudo, pseudo=pseudo).y.format('concat_'))

  # Limit number of points plotted.
  PCA_1 = PCA_1[:M]
  PCA_2 = PCA_2[:M]
  y = y[:M]
  return PCA_1, PCA_2, y

################################################################################
# Plot.                                                                        #
################################################################################

def plot(ps_PCA_1, ps_PCA_2, ps_y, PCA_1, PCA_2, y):
  # Start figure.
  fig = plt.figure()
  ax  = fig.add_axes([0.15, 0.15, 0.80, 0.80])

  # Plot.
  for latt in cnst.lattices:
    y_lbl = latt.y_label
    ax.plot(PCA_1[y==y_lbl],       PCA_2[y==y_lbl]      , latt.pt_fmt,    marker='.', alpha=.3, ls='', label=latt.name, mew=0)
    ax.plot(ps_PCA_1[ps_y==y_lbl], ps_PCA_2[ps_y==y_lbl], latt.ps_pt_fmt, marker='.', alpha=.3, ls='', label='pseudo '+latt.name, mew=0)
   
  # Add details.
  ax.set_xlabel(r'First PCA component')
  ax.set_ylabel(r'Second PCA component')
  ax.legend()

  # Save figure.
  fig.savefig(dir_util.pca_fig_path03(), dpi=300)
  plt.close()

################################################################################

def main():
  ps_PCA_1, ps_PCA_2, ps_y = load_and_process_data(pseudo=True)
  PCA_1, PCA_2, y          = load_and_process_data(pseudo=False)
  plot(ps_PCA_1, ps_PCA_2, ps_y, PCA_1, PCA_2, y)

if __name__=='__main__':
  main()

