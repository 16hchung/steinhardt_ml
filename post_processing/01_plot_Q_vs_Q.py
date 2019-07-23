import sys
sys.path.append('..')
from auxiliary import *

import matplotlib.pyplot as plt                 
from numpy import *
c = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#FFFF33', '#A65628', '#F781BF', '#999999']


################################################################################
# Load and process data.                                                       #
################################################################################

dr_oct = loadtxt('../data/random_structures/dr_0.dat')
dr_tet = loadtxt('../data/random_structures/dr_1.dat')
dr_tri = loadtxt('../data/random_structures/dr_2.dat')
dr_sqr = loadtxt('../data/random_structures/dr_3.dat')

# Panel parameters.
N = 10
dL0_x = 0.08
dL0_y = 0.07
L = 1/(N+1)
dL = 0.90*L

marker = ['8','D', '^', 's']

################################################################################
# Plot Q vs l.                                                                 #
################################################################################

# Start figure.
fig = plt.figure()
ax  = fig.add_axes([0.11,0.15,0.8,0.8])

# Plot.
for l in range(N):
  plt.axhline(y=l+1, ls='--', c='k', lw=0.5)
  for y in range(4):
    Q = loadtxt('../data/random_structures/Q_%d.dat' % y)
    ax.plot(Q[:,l], (l+1)*ones(Q.shape[0]), marker[y], c=c[y], ms=6, alpha=0.5)
    ax.plot(Q0[y,l], (l+1), marker[y], c='k', ms=6)

# Add details.
ax.set_xlabel(r'$Q_\ell$', fontsize=20)
ax.set_ylabel(r'$\ell$', fontsize=20)
ax.set_yticks(arange(1,10+1))

# Save figure.
fig.savefig("figures/Q/fig_Q_vs_l.png", dpi=300)
plt.close()

################################################################################
# Plot Q_4 vs Q_6.                                                             #
################################################################################

# Start figure.
fig = plt.figure()
ax  = fig.add_axes([0.12,0.15,0.8,0.8])

# Plot.
for y in range(4):
  Q = loadtxt('../data/random_structures/Q_%d.dat' % y)
  ax.plot(Q[:,3], Q[:,5], marker[y], c=c[y], ms=7, alpha=0.5)
  ax.plot(Q0[y,3], Q0[y,5], marker[y], c='k', ms=7)

# Add details.
ax.set_xlabel(r'$Q_4$', fontsize=20)
ax.set_ylabel(r'$Q_6$', fontsize=20)

# Save figure.
fig.savefig("figures/Q/fig_Q4_vs_Q6.png", dpi=300)
plt.close()

################################################################################
# Plot panel.                                                                  #
################################################################################

fig = plt.figure(figsize=(15,15))
for li in range(N):
  for lj in range(N):
    ax  = fig.add_axes([dL0_x+li*L, dL0_y+lj*L, dL, dL])
    for y in range(4):
      Q = loadtxt('../data/random_structures/Q_%d.dat' % y)
      ax.plot(Q[:,li], Q[:,lj], marker[y], c=c[y], ms=6)
      ax.plot(Q0[y,li], Q0[y,lj], marker[y], c='k', ms=6)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.tick_params(axis='y', which='both', right=False, left=False)
    if li == 0:
      ax.annotate(r'$Q_{%d}$' % (lj+1), size=30, xy=(-0.65,0.40), xycoords='axes fraction')
    if lj == 0:
      ax.annotate(r'$Q_{%d}$' % (li+1), size=30, xy=(0.33,-0.40), xycoords='axes fraction')
fig.savefig("figures/Q/fig_Q_vs_Q.png", dpi=300)
plt.close()

################################################################################
