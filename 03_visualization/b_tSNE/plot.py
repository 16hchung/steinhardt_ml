import matplotlib.pyplot as plt              
import numpy as np

from .compute import perplexity_list
from util import constants as cnst
from util import dir_util

# Input data.
#y = loadtxt('../data/y.dat')
#perplexity_list = [10, 50, 200, 500, 1000]

################################################################################
# Plot tSNE.                                                                   #
################################################################################

#for perplexity in perplexity_list:
#  # Load results.
#  tSNE = loadtxt('../data/tSNE_%d.dat' % perplexity)
#  
#  # Start figure.
#  fig = plt.figure()
#  ax  = fig.add_axes([0.025, 0.025, 0.925, 0.925])
#  
#  # Plot.
#  ax.plot(tSNE[y==0][:,0], tSNE[y==0][:,1], 'C3o', alpha=0.3, label='fcc', mew=0)
#  ax.plot(tSNE[y==1][:,0], tSNE[y==1][:,1], 'C0s', alpha=0.3, label='bcc', mew=0)
#  ax.plot(tSNE[y==2][:,0], tSNE[y==2][:,1], 'C2^', alpha=0.3, label='hcp', mew=0)
#   
#  # Add details.
#  ax.set_xticks([])
#  ax.set_yticks([])
#  ax.legend()
#  
#  # Save figure.
#  fig.savefig("figures/fig_tSNE_%d.png" % perplexity, dpi=300)
#  plt.close()
def plot_one_tSNE(X_tmplt, fig_tmplt, y, perplexity):
  # Load results.
  tSNE = np.loadtxt(X_tmplt.format(perplexity))
  
  # Start figure.
  fig = plt.figure()
  ax  = fig.add_axes([0.025, 0.025, 0.925, 0.925])
  
  # Plot.
  for latt in cnst.lattices:
    y_lbl = latt.y_label
    ax.plot(tSNE[y==y_lbl][:,0], tSNE[y==y_lbl][:,1], latt.pt_fmt, alpha=.3, label=latt.name, mew=0)
   
  # Add details.
  ax.set_xticks([])
  ax.set_yticks([])
  ax.legend()
  
  # Save figure.
  fig.savefig(fig_tmplt.format(perplexity), dpi=300)
  plt.close()

def plot_many_tSNE(y, data_paths, withPCA=False):
  X_tmplt = data_paths.X_with_PCA_tmplt if withPCA else data_paths.X_tmplt
  fig_tmplts = dir_util.tSNE_fig_tmplts03()
  fig_tmplt = fig_tmplts.with_PCA if withPCA else fig_tmplts.no_PCA
  
  for perplexity in perplexity_list:
    plot_one_tSNE(X_tmplt, fig_tmplt, y, perplexity)

################################################################################
# Plot tSNE after PCA filter.                                                  #
################################################################################

#for perplexity in perplexity_list:
#  # Load results.
#  tSNE = loadtxt('../data/tSNE_PCA_%d.dat' % perplexity)
#  
#  # Start figure.
#  fig = plt.figure()
#  ax  = fig.add_axes([0.025, 0.025, 0.925, 0.925])
#  
#  # Plot.
#  ax.plot(tSNE[y==0][:,0], tSNE[y==0][:,1], 'C3o', alpha=0.3, label='fcc', mew=0)
#  ax.plot(tSNE[y==1][:,0], tSNE[y==1][:,1], 'C0s', alpha=0.3, label='bcc', mew=0)
#  ax.plot(tSNE[y==2][:,0], tSNE[y==2][:,1], 'C2^', alpha=0.3, label='hcp', mew=0)
#   
#  # Add details.
#  ax.set_xticks([])
#  ax.set_yticks([])
#  ax.legend()
#  
#  # Save figure.
#  fig.savefig("figures/fig_tSNE_PCA_%d.png" % perplexity, dpi=300)
#  plt.close()

################################################################################
def main():
  data_paths = dir_util.tSNE_data_paths03()
  y = np.loadtxt(data_paths.y)
  plot_many_tSNE(y, data_paths, withPCA=False)
  plot_many_tSNE(y, data_paths, withPCA=True)

if __name__=='__main__':
  main()

