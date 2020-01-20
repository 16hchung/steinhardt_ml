import matplotlib.pyplot as plt                 
import numpy as np

from util import dir_util
from util import constants as cnst

################################################################################
# Load and process data.                                                       #
################################################################################

def load_and_process_data():
  M = 20000 # Number of points to be plotted.

  # Load results.
  data_names = dir_util.pca_data_paths03()
  PCA_1 = np.loadtxt(data_names.comp1)
  PCA_2 = np.loadtxt(data_names.comp2)
  y = np.loadtxt(dir_util.clean_features_paths02(istest=True).y)

  # Limit number of points plotted.
  PCA_1 = PCA_1[:M]
  PCA_2 = PCA_2[:M]
  y = y[:M]
  return PCA_1, PCA_2, y

################################################################################
# Plot.                                                                        #
################################################################################

def plot(PCA_1, PCA_2, y):
  # Start figure.
  fig = plt.figure()
  ax  = fig.add_axes([0.15, 0.15, 0.80, 0.80])

  # Plot.
  for latt in cnst.lattices:
    y_lbl = latt.y_label
    ax.plot(PCA_1[y==y_lbl], PCA_2[y==y_lbl], latt.pt_fmt, alpha=.3, label=latt.name, mew=0)
   
  # Add details.
  ax.set_xlabel(r'First PCA component')
  ax.set_ylabel(r'Second PCA component')
  ax.legend()

  # Save figure.
  fig.savefig(dir_util.pca_fig_path03(), dpi=300)
  plt.close()

################################################################################

def main():
  PCA_1, PCA_2, y = load_and_process_data()
  plot(PCA_1, PCA_2, y)

if __name__=='__main__':
  main()

