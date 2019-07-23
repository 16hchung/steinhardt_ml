from auxiliary import *

from numpy import *
from numpy.random import random
from scipy.linalg import norm
from scipy.spatial.transform import Rotation

################################################################################
# Auxiliary functions.
################################################################################

# Randomly rotate r.
def random_rotation(r):
  alpha = 2*pi*random()
  beta = 2*pi*random()
  gamma = 2*pi*random()
  Rx = Rotation.from_euler('x', alpha).as_dcm()
  Ry = Rotation.from_euler('y', beta).as_dcm()
  Rz = Rotation.from_euler('z', gamma).as_dcm()
  R = matmul(Rx,matmul(Ry,Rz))
  for i in range(r.shape[0]):
    r[i] = matmul(R,r[i])
  return r

# Compute Steinhard for random distortions.
def compute_distortion(label,r,dr_max,l,N):
  dr_total = zeros(N) # Total distortion.
  with open('data/random_structures/Q_%d.dat' % label,'w') as f:
    for n in range(N):
      # Create a random distortion of the structure.
      dr = random(r.shape)-0.5 # Distortion vector.
      for i in range(dr.shape[0]):
        dr[i] = dr_max * random() * dr[i]/norm(dr[i])
      # Select 20% of the distortions to be a pure shear of the ideal structure.
      if n < int(0.2*N):
        for i in range(1,dr.shape[0]):
          dr[i] = dr[0]
      dr_total[n] = sqrt(sum(norm(dr,axis=1)**2))
      # Compute and save Steinhardt paramter.
      for Q in steinhardt(random_rotation(r+dr),l):
        f.write('%.10f ' % Q)
      f.write('\n')
  savetxt('data/random_structures/dr_%d.dat' % label, dr_total)

################################################################################
# Compute.
################################################################################

dr_max = 0.2 # Distortion amplitude
l = arange(1,10+1) # Range of spherical harmonics used.
N = 1000 # Number of samples for each structure.

# Compute Q and distortion for each structure.
for y in range(r0.shape[0]):
  compute_distortion(y,r0[y],dr_max,l,N)

################################################################################
