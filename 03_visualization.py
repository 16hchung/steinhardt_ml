from auxiliary import *

from numpy import *
from numpy.random import choice
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.externals import joblib

# Load data.
X = loadtxt('data/classifiers/X.dat')
y = loadtxt('data/classifiers/y.dat')

# PCA analysis on the distorted structures.
pca = PCA().fit(X)
X_pca = pca.transform(X)
var = pca.explained_variance_ratio_.cumsum()
savetxt('data/tsne_pca/pca_explained_variance.dat', var)
savetxt('data/tsne_pca/pca.dat', X_pca)

# PCA analysis on the ideal structures.
scaler = joblib.load('data/classifiers/scaler.pkl')
X0_pca = pca.transform(scaler.transform(Q0))
savetxt('data/tsne_pca/pca0.dat', X0_pca)

# tSNE.
pca = PCA(n_components=5)
X_pca = pca.fit(X).transform(X)
small_set = choice(X.shape[0], size=300, replace=False)
savetxt('data/tsne_pca/y_tsne_set.dat', y[small_set])
for perplexity in [10,20,50,100]:
  print(perplexity)
  X_tsne = TSNE(perplexity=perplexity).fit_transform(X_pca[small_set])
  savetxt('data/tsne_pca/tsne_%d.dat' % perplexity, X_tsne)
