from auxiliary import *

from numpy import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

################################################################################
# Train SVM classifier and scaler.                                             #
################################################################################

# Load features.
X_oct = loadtxt('data/random_structures/Q_0.dat')
X_tet = loadtxt('data/random_structures/Q_1.dat')
X_tri = loadtxt('data/random_structures/Q_2.dat')
X_sqr = loadtxt('data/random_structures/Q_3.dat')

# Create labels.
y_oct = 0*ones(X_oct.shape[0],dtype=int)
y_tet = 1*ones(X_tet.shape[0],dtype=int)
y_tri = 2*ones(X_tri.shape[0],dtype=int)
y_sqr = 3*ones(X_sqr.shape[0],dtype=int)

# Concatenate features and labels. 
X = row_stack((X_oct,X_tet,X_tri,X_sqr))
y = array(concatenate((y_oct,y_tet,y_tri,y_sqr)), dtype=int)

# Scale and shuffle features.
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
X,y = shuffle(X,y)

# Train SVM classifier.
clf = LinearSVC(C=1.0)
clf.fit(X,y)

# Save data used to train scaler and SVM. Also saves the classifiers.
savetxt('data/classifiers/y.dat',y)
savetxt('data/classifiers/X.dat',X)
joblib.dump(scaler,'data/classifiers/scaler.pkl')
joblib.dump(clf,'data/classifiers/svm.pkl')

################################################################################
# Train PCA.
################################################################################

for y in range(4):
  X = loadtxt('data/random_structures/Q_%d.dat' % y)
  X = scaler.transform(X)
  pca = PCA().fit(X)
  d = zeros(X.shape[0])
  for n in range(X.shape[0]):
    d[n] = distortion(X[n],y,pca,scaler)
  X = pca.transform(X)
  savetxt('data/distortion_pca/pca_%d.dat' % y, X)
  savetxt('data/distortion_pca/d_%d.dat' % y, d)
  savetxt('data/distortion_pca/pca0_%d.dat' % y, pca.transform(scaler.transform(Q0[y].reshape(1,-1))))
  joblib.dump(pca,'data/classifiers/pca_%d.pkl' % y)

################################################################################
