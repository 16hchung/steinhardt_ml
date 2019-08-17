import matplotlib.pyplot as plt                 
from numpy import *
c = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#FFFF33', '#A65628', '#F781BF', '#999999']

################################################################################
# Setup.                                                                       #
################################################################################

# List of structures.
y_to_str = ['fcc','bcc','hcp','liq']
marker = ['8','D', '^', 's']

################################################################################
# Plot PCA distance vs dr.                                                     #
################################################################################

# Start figure.
fig = plt.figure()
ax  = fig.add_axes([0.14, 0.12, 0.85, 0.85])

# Plot
y, dr, d = loadtxt('../data/test/distortion.dat', unpack=True)
for i in range(len(marker)):
  ax.plot(d[y==i], dr[y==i], marker[i], c=c[i], alpha=0.4, ms=2)
   
# Add details and save figure.
ax.set_xlim(0,13)
ax.set_ylim(0)
ax.set_xlabel(r'$\sqrt{[\Delta (PCA_1)]^2 + [\Delta (PCA_2)]^2}$ ')
ax.set_ylabel(r'$\sqrt{\sum_i(\Delta \mathbf{r}_i)^2}$')
fig.savefig("figures/fig_d_vs_dr_test.png", dpi=300)
plt.close()

################################################################################
