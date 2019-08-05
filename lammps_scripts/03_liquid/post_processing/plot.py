import matplotlib.pyplot as plt                 
from numpy import *

################################################################################
# Load and process data.                                                       #
################################################################################

lattices = ['fcc', 'bcc', 'hcp']

T_target = 0.92
P_target = 5.68

################################################################################
# Plot temperature.                                                            #
################################################################################

# Start figure.
fig = plt.figure()
ax  = fig.add_axes([0.15, 0.15, 0.80, 0.80])

# Plot crystals.
for i in range(3):
  t, T, P, rho = loadtxt('../../02_crystals/data/thermo_%s.dat' % lattices[i], unpack=True)
  ax.plot(t, T, 'C%d-' % i, label=lattices[i], alpha=0.5)
  plt.axhline(T[100:].mean(), ls='--', c='C%d' % i, lw=1, zorder=999)

# Plot liquid.
t, T, P, rho = loadtxt('../data/thermo_liquid.dat', unpack=True)
ax.plot(t, T, 'C3-', label='liquid', alpha=0.5)
plt.axhline(T[100:].mean(), ls='--', c='C3', lw=1, zorder=999)
plt.axhline(T_target, ls='--', c='k', lw=0.5, label='target')
 
# Add details and save figure.
ax.set_xlabel(r'time')
ax.set_ylabel(r'temperature')
ax.set_xlim(0)
ax.legend()

# Save figure.
fig.savefig("figures/fig_temperature.png", dpi=300)
plt.close()

################################################################################
# Plot pressure.                                                               #
################################################################################

# Start figure.
fig = plt.figure()
ax  = fig.add_axes([0.15, 0.15, 0.80, 0.80])

# Plot.
for i in range(3):
  t, T, P, rho = loadtxt('../../02_crystals/data/thermo_%s.dat' % lattices[i], unpack=True)
  ax.plot(t, P, 'C%d-' % i, label=lattices[i], alpha=0.5)
  plt.axhline(P[100:].mean(), ls='--', c='C%d' % i, lw=1, zorder=999)

# Plot liquid.
t, T, P, rho = loadtxt('../data/thermo_liquid.dat', unpack=True)
ax.plot(t, P, 'C3-', label='liquid', alpha=0.5)
plt.axhline(P[100:].mean(), ls='--', c='C3', lw=1, zorder=999)
plt.axhline(P_target, ls='--', c='k', lw=0.5, label='target')
 
# Add details and save figure.
ax.set_xlabel(r'time')
ax.set_ylabel(r'pressure')
ax.set_xlim(0)
ax.legend()

# Save figure.
fig.savefig("figures/fig_pressure.png", dpi=300)
plt.close()

################################################################################
# Plot density.                                                                #
################################################################################

# Start figure.
fig = plt.figure()
ax  = fig.add_axes([0.15, 0.15, 0.80, 0.80])

# Plot.
for i in range(3):
  t, T, P, rho = loadtxt('../../02_crystals/data/thermo_%s.dat' % lattices[i], unpack=True)
  ax.plot(t, rho, 'C%d-' % i, label=lattices[i], alpha=0.5)
  plt.axhline(rho[100:].mean(), ls='--', c='C%d' % i, lw=1, zorder=999)

# Plot liquid.
t, T, P, rho = loadtxt('../data/thermo_liquid.dat', unpack=True)
ax.plot(t, rho, 'C3-', label='liquid', alpha=0.5)
plt.axhline(rho[100:].mean(), ls='--', c='C3', lw=1, zorder=999)
 
# Add details and save figure.
ax.set_xlabel(r'time')
ax.set_ylabel(r'density')
ax.set_xlim(0)
ax.legend()

# Save figure.
fig.savefig("figures/fig_density.png", dpi=300)
plt.close()

################################################################################
