from auxiliary import *

from numpy import *
from numpy.random import random, randint
from sklearn.preprocessing import scale
from scipy.special import sph_harm
from scipy.linalg import norm
from sklearn.externals import joblib
from sklearn.metrics import classification_report

################################################################################
# Setup.                                                                       #
################################################################################

# Input paramters.
dr_max = 0.2 # Distortion amplitude.
l = arange(1,10+1) # Range of spherical harmonics used.
N = 4000 # Number of samples.

# Computed quantities.
y = zeros(N,dtype=int) # Labels.
dr_total = zeros(N) # Total distortion.
y_pred = zeros(N,dtype=int) # Predicted labels.
d = zeros(N) # PCA distortion.

#  Load scaler, SVM, and PCAs.
scaler = joblib.load('data/classifiers/scaler.pkl')
svm = joblib.load('data/classifiers/svm.pkl')
pca = []
for i in range(4):
  pca.append(joblib.load('data/classifiers/pca_%d.pkl' % i))

################################################################################
# Test SVM accuracy.                                                           #
################################################################################

for n in range(N):
  # Randomly select one structure
  y[n] = randint(4)
  # Create random distortion (or shear).
  dr = random(r0[y[n]].shape)-0.5
  for i in range(dr.shape[0]):
    dr[i] = dr_max * random() * dr[i]/norm(dr[i])
  if random() <= 0.2:
    # Create a random shear of the structure.
    for i in range(dr.shape[0]):
      dr[i] = dr[0]
  dr_total[n] = sqrt(sum(norm(dr,axis=1)**2)) # Compute total distortion.

  # Compute structure and PCA distortion.
  y_pred[n], d[n] = compute_structural_features(r0[y[n]]+dr,l,svm,pca,scaler)

# Save distortion data.
savetxt('data/test/distortion.dat', column_stack((y,dr_total,d)), header='y | dr [A] | d_pca')

# Test SVM precision.
with open('data/test/classification_report.dat', 'w') as f:
  f.write(classification_report(y,y_pred))
  print(classification_report(y,y_pred))

################################################################################
