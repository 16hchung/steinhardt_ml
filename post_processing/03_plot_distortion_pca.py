import matplotlib.pyplot as plt                 
from numpy import *
c = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#FFFF33', '#A65628', '#F781BF', '#999999']

################################################################################
# Setup.                                                                       #
################################################################################

# List of structures.
y_to_str = ['octahedral', 'tetrahedral', 'trigonal_prismatic', 'square']
marker = ['8','D', '^', 's']

################################################################################
# Plot PCA per structure.                                                      #
################################################################################

for y in range(4):
  X = loadtxt('../data/distortion_pca/pca_%d.dat' % y)
  X0 = loadtxt('../data/distortion_pca/pca0_%d.dat' % y)
  dr = loadtxt('../data/random_structures/dr_%d.dat' % y)
  
  # Start figure.
  fig = plt.figure()
  ax  = fig.add_axes([0.12, 0.12, 0.85, 0.85])
  plt.axhline(ls='--', c='k', lw=0.5)
  plt.axvline(ls='--', c='k', lw=0.5)
  
  # Plot.
  ax.scatter(X[:,0], X[:,1], marker=marker[y], c=dr, cmap='winter', alpha=0.5, lw=0)
  ax.scatter(X0[0], X0[1], marker=marker[y], c='k')
   
  # Add details and save figure.
  ax.set_xlabel(r'PCA 1')
  ax.set_ylabel(r'PCA 2')
  fig.savefig("figures/distortion/fig_%s.png" % y_to_str[y], dpi=300)
  plt.close()

################################################################################
# Plot PCA distance vs dr.                                                     #
################################################################################

# Start figure.
fig = plt.figure()
ax  = fig.add_axes([0.14, 0.12, 0.85, 0.85])

for i in range(len(marker)):
  d = loadtxt('../data/distortion_pca/d_%d.dat' % i)
  dr = loadtxt('../data/random_structures/dr_%d.dat' % i)
  ax.plot(d, dr, marker[i], c=c[i], alpha=0.4)
  ax.plot(0, 0, marker[i], c='k')
   
# Add details and save figure.
ax.set_xlim(0,13)
ax.set_ylim(0)
ax.set_xlabel(r'$\sqrt{[\Delta (PCA_1)]^2 + [\Delta (PCA_2)]^2}$ ')
ax.set_ylabel(r'$\sqrt{\sum_i(\Delta \mathbf{r}_i)^2}$')
fig.savefig("figures/distortion/fig_d_vs_dr.png", dpi=300)
plt.close()

################################################################################
