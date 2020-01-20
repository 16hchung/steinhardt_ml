import matplotlib.pyplot as plt                 
import numpy as np

from util import constants as cnst, dir_util

################################################################################
# Setup.                                                                       #
################################################################################
# Constants
M = 2000 # Number of points to be plotted.

# Panel parameters.
N = 10 # Number of ell values considered (from 1 to N).
dL0_x = 0.08 # Initial spacing along x.
dL0_y = 0.10 # Initial spacing along y.
L = 1/(N+1) # Total length occupied by panels in each direction.
dL = 0.90*L # Length of panel sides.

def load_data():
  paths = dir_util.clean_features_paths02(pseudo=True)
  X = np.loadtxt(paths.unscaledX)
  y = np.loadtxt(paths.y)
  X = X[:M]
  y = y[:M]
  return X, y

################################################################################
# Plot unscaled data.                                                          #
################################################################################

def main():
  X,y = load_data()

  # Start figure.
  fig = plt.figure(figsize=(15,15))

  # Loop over all unique ell pairs.
  for li in range(N):
    for lj in range(li,N):
      ax  = fig.add_axes([dL0_x+li*L, 1-dL0_y-lj*L, dL, dL]) # Create frames.

      for latt in cnst.lattices:
        y_lbl = latt.y_label
        ax.plot(X[y==y_lbl][:,li], X[y==y_lbl][:,lj], latt.pt_fmt, alpha=.1, mew=0)
      
      # Remove panels and ticks.
      ax.set_xticklabels([])
      ax.set_yticklabels([])
      ax.tick_params(axis='x', which='both', bottom=False, top=False)
      ax.tick_params(axis='y', which='both', right=False, left=False)

      # Add panel labels.
      if li == 0:
        ax.annotate(r'$Q_{%d}$' % (lj+1), size=30, xy=(-0.65,0.40), xycoords='axes fraction')
      if lj == N-1:
        ax.annotate(r'$Q_{%d}$' % (li+1), size=30, xy=(0.33,-0.40), xycoords='axes fraction')
  fig.savefig(dir_util.feat_pairs_map_path03(), dpi=300)
  plt.close()

################################################################################
if __name__=='__main__':
  main()
