from numpy import *
from scipy.linalg import norm
from scipy.special import sph_harm

################################################################################
# Auxiliary functions.                                                         #
################################################################################

# Compute Steinhardt parameter.
def steinhardt(r,l):
  Q = zeros(len(l))
  for i in range(len(l)):
    q = zeros(2*l[i]+1,dtype=complex)
    for v in r:
      phi = arctan2(v[1],v[0])
      theta = arccos(v[2]/norm(v))
      q += sph_harm(arange(-l[i],l[i]+1),l[i],phi,theta)
    q /= r.shape[0]
    Q[i] = sqrt(real((4*pi/(2*l[i]+1)) * sum(q*conjugate(q))))
  return Q

# Compute distortion of X. X must have been normalized by scaler.
def distortion(X,y,pca,scaler):
  X = pca.transform(X.reshape(1,-1))[0]
  X0 = pca.transform(scaler.transform(Q0[y].reshape(1,-1)))[0]
  d = sqrt((X[0]-X0[0])**2 + (X[1]-X0[1])**2)
  return d

# Compute structural features (class and distortion).
def compute_structural_features(r,l,clf,pca,scaler):
  Q = steinhardt(r,l)
  X = scaler.transform(Q.reshape(1,-1))
  y = int(clf.predict(X.reshape(1,-1)))
  d = distortion(X,y,pca[y],scaler)
  return y, d

################################################################################
# Geometry of perfect structures.                                              #
################################################################################

# Octahedral geometry.
r_oct = array([[ 1, 0, 0],
               [-1, 0, 0],
               [ 0, 1, 0],
               [ 0,-1, 0],
               [ 0, 0, 1],
               [ 0, 0,-1]])

# Tetrahedral geometry.
r_tet = array([[0,                  0,   1],
               [0,          sqrt(8/9),-1/3],
               [-sqrt(2/3),-sqrt(2/9),-1/3],
               [+sqrt(2/3),-sqrt(2/9),-1/3]])

# Trigonal prismatic geometry.
r_tri = array([[0,          sqrt(8/9),-1/3],
               [-sqrt(2/3),-sqrt(2/9),-1/3],
               [+sqrt(2/3),-sqrt(2/9),-1/3],
               [0,          sqrt(8/9),+1/3],
               [-sqrt(2/3),-sqrt(2/9),+1/3],
               [+sqrt(2/3),-sqrt(2/9),+1/3]])

# Planar square geometry.
r_sqr = array([[ 1, 0, 0],
               [-1, 0, 0],
               [ 0, 1, 0],
               [ 0,-1, 0]])

# Collection of all geometries.
r0 = array([r_oct, r_tet, r_tri, r_sqr])

# Dictionary and lists linking structures and labels.
y_to_str = ['octahedral', 'tetrahedral', 'trigonal_prismatic', 'square']
str_to_y = {'octahedral': 0, 
            'tetrahedral': 1, 
            'trigonal_prismatic': 2,
            'square': 3}

# Steinhardt parameter of perfect structures.
Q0 = zeros((r0.shape[0],10))
for y in range(r0.shape[0]):
  Q0[y] = steinhardt(r0[y],arange(1,10+1))

################################################################################
