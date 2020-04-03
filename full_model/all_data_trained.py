from sklearn.svm import LinearSVC
import numpy as np

from util import constants as cnst

class AllDataLinearSVC(LinearSVC):
  def __init__(self, *kargs, **kwargs):
    super().__init__(*kargs, **kwargs)

  def predict(self, X):
    y = super().predict(X)
    y[y<0] = -1
    return y

class AllDataLinearSVCLayered(LinearSVC):
  def __init__(self, n_possible_neigh=2, *kargs, **kwargs):
    super().__init__(*kargs, **kwargs)
    self.n_possible_neigh = n_possible_neigh

  def predict(self, X):
    n_feat = int(X.shape[1] / self.n_possible_neigh)
    ys = []
    y = None
    for i in range(self.n_possible_neigh):
      currX = X[:, i*n_feat : (i+1)*n_feat]
      y = super().predict(currX)
      ys.append(y)
    y[(ys[0] < 0) | (ys[1] < 0)] = -1
    y[(ys[0] != ys[1]) & (ys[0] > 0) & (ys[1] > 0)] = -2
    return y
