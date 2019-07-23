import matplotlib.pyplot as plt                 
from numpy import *
c = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#FFFF33', '#A65628', '#F781BF', '#999999']

################################################################################
# Load and process data.                                                       #
################################################################################

y = loadtxt('../data/classifiers/y.dat')
X_pca = loadtxt('../data/tsne_pca/pca.dat')
X0_pca = loadtxt('../data/tsne_pca/pca0.dat')

marker = ['8','D', '^', 's']

################################################################################
# Plot PCA.                                                                    #
################################################################################

# Start figure.
fig = plt.figure()
ax  = fig.add_axes([0.12, 0.12, 0.85, 0.85])
plt.axhline(ls='--', c='k', lw=0.5)
plt.axvline(ls='--', c='k', lw=0.5)

# Plot.
for i in range(4):
  ax.plot(X_pca[y==i,0], X_pca[y==i,1], marker[i], c=c[i], ms=6, alpha=0.5)
  ax.plot(X0_pca[i,0], X0_pca[i,1], marker[i], c='k', ms=6)
 
# Add details and save figure.
ax.set_xlabel(r'PCA 1')
ax.set_ylabel(r'PCA 2')
fig.savefig("figures/visualization/fig_PCA.png", dpi=300)
plt.close()

################################################################################
# Plot tSNE.                                                                   #
################################################################################

y = loadtxt('../data/tsne_pca/y_tsne_set.dat')
for perplexity in [10,20,50,100]:
  X_tsne = loadtxt('../data/tsne_pca/tsne_%d.dat' % perplexity)
  fig = plt.figure()
  ax  = fig.add_axes([0.05, 0.05, 0.90, 0.90])
  for i in range(4):
    ax.plot(X_tsne[y==i,0], X_tsne[y==i,1], marker[i], c=c[i], ms=6)
  ax.set_xticks([])
  ax.set_yticks([])
  fig.savefig("figures/visualization/fig_tSNE_%d.png" % perplexity, dpi=300)
  plt.close()

################################################################################
