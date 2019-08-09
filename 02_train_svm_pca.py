from auxiliary import *
from softmax import Softmax

from numpy import *
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

################################################################################
# Train SVM classifier and scaler.                                             #
################################################################################

# Load features.
X_fcc = loadtxt('data/from_sim/fcc_steinhardt.dat')
X_bcc = loadtxt('data/from_sim/bcc_steinhardt.dat')
X_hcp = loadtxt('data/from_sim/hcp_steinhardt.dat')
X_liq = loadtxt('data/from_sim/liq_steinhardt.dat')

# Create labels.
y_fcc = 0*ones(X_fcc.shape[0],dtype=int)
y_bcc = 1*ones(X_bcc.shape[0],dtype=int)
y_hcp = 2*ones(X_hcp.shape[0],dtype=int)
y_liq = 3*ones(X_liq.shape[0],dtype=int)

# Concatenate features and labels. 
X = row_stack((X_fcc,X_bcc,X_hcp,X_liq))
y = array(concatenate((y_fcc,y_bcc,y_hcp,y_liq)), dtype=int)

n = len(y)
partition = int(n * .8)
X,y = shuffle(X,y)
X_train = X[:partition][:]
y_train = y[:partition]
X_val = X[partition:][:]
y_val = y[partition:]

# Scale and shuffle features.
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

# Train SVM classifier.
clf = LinearSVC(C=1.0, max_iter=2000)
clf.fit(X_train, y_train)
print("linear score on val set: ", clf.score(X_val, y_val))
 
# Nonlinear SVM
clf2 = SVC(C=1.0, max_iter=1000)
clf2.fit(X_train, y_train)
print("kernelized score on val set: ", clf2.score(X_val, y_val))

# Softmax
clf3 = Softmax()
clf3.train(X_train, y_train)
y_pred = clf3.predict(X_val)
print("softmax score on val set: ", mean(equal(y_val, y_pred)))

# Save data used to train scaler and SVM. Also saves the classifiers.
savetxt('data/classifiers/y.dat',y_train)
savetxt('data/classifiers/X.dat',X_train)
joblib.dump(scaler,'data/classifiers/scaler.pkl')
joblib.dump(clf,'data/classifiers/svm.pkl')

################################################################################
# Train PCA.
################################################################################

#for y, X in enumerate([X_fcc, X_bcc, X_hcp, X_liq]):
#  X = scaler.transform(X)
#  pca = PCA().fit(X)
#  d = zeros(X.shape[0])
#  for n in range(X.shape[0]):
#    d[n] = distortion(X[n],y,pca,scaler)
#  X = pca.transform(X)
#  savetxt('data/distortion_pca/pca_%d.dat' % y, X)
#  savetxt('data/distortion_pca/d_%d.dat' % y, d)
#  savetxt('data/distortion_pca/pca0_%d.dat' % y, pca.transform(scaler.transform(Q0[y].reshape(1,-1))))
#  joblib.dump(pca,'data/classifiers/pca_%d.pkl' % y)

################################################################################
