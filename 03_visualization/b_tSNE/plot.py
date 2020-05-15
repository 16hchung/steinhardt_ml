import matplotlib.pyplot as plt              
import numpy as np

from .compute import perplexity_list, split_to_real_pseudo
from util import constants as cnst
from util import dir_util


################################################################################
# Plot tSNE.                                                                   #
################################################################################

def plot_one_tSNE(fig_name, tSNE, y):
  # Start figure.
  fig = plt.figure()
  ax  = fig.add_axes([0.025, 0.025, 0.925, 0.925])
  
  # Plot.
  for latt in cnst.lattices:
    y_lbl = latt.y_label * -1 # TODO remove hack
    ax.plot(tSNE[y==y_lbl][:,0],       tSNE[y==y_lbl][:,1],       latt.pt_fmt,    marker='o', alpha=.3, ls='', label=latt.name+' liq', mew=0)
   
  # TODO remove hack
  ax.plot(tSNE[y==2][:,0], tSNE[y==2][:,1], 'ko', marker = 'o', alpha=.3, ls='', label='hcp', mew=0)

  # Add details.
  ax.set_xticks([])
  ax.set_yticks([])
  ax.legend()
  
  # Save figure.
  fig.savefig(fig_name, dpi=300)
  plt.close()

def plot_many_tSNE(ps_y, y, data_paths, withPCA=False):
  X_tmplt = data_paths.X_with_PCA_tmplt if withPCA else data_paths.X_tmplt
  fig_tmplts = dir_util.tSNE_fig_tmplts03()
  fig_tmplt = fig_tmplts.with_PCA if withPCA else fig_tmplts.no_PCA
  
  for perplexity in perplexity_list:
    tSNE = np.loadtxt(X_tmplt.format(perplexity))
    #tSNE, ps_tSNE = split_to_real_pseudo(tSNE)
    plot_one_tSNE(fig_tmplt.format(perplexity), tSNE, y) # TODO vvv remove hack
    #plot_one_tSNE(fig_tmplt.format(str(perplexity)+'synth'), ps_tSNE, ps_y)

################################################################################
# Plot tSNE after PCA filter.                                                  #
################################################################################

def main():
  data_paths    = dir_util.tSNE_data_paths03()
  ps_data_paths = dir_util.tSNE_data_paths03(pseudo=True)
  y    = np.loadtxt(data_paths.y)
  #ps_y = np.loadtxt(ps_data_paths.y) TODO remove hack
  plot_many_tSNE(y, y, data_paths, withPCA=False)
  #plot_many_tSNE(ps_y, y, data_paths, withPCA=False)
  #plot_many_tSNE(ps_y, y, data_paths, withPCA=True)

if __name__=='__main__':
  main()

