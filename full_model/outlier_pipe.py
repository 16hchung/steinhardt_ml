from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from tqdm import tqdm

from util.const_perf import perf_features
import util.constants as cnst

class ClassifierWithLiq:
  def __init__(self, outlier=OneClassSVM, classifier=SVC, outlier_args={'nu':.05}, class_args={}):
    dflt_outlier = {'nu':.05}
    dflg_classif = {}
    dflt_outlier.update(outlier_args)
    dflg_classif.update(class_args)
    self.classifier = classifier(**dflg_classif)
    self.outlier    = outlier(**dflt_outlier)

  def fit(self, X, y):
    self.classifier.fit(X, y)
    self.outlier.fit(X)
    return self

  def get_params(self, *kargs, **kwargs):
    return self.classifier.get_params(*kargs, **kwargs)

  def set_params(self, *kargs, **kwargs):
    return self.classifier.set_params(*kargs, **kwargs)

  def predict(self, X):
    y_pre = self.outlier.predict(X)
    y = self.classifier.predict(X)
    y[y_pre < 0] = -1
    return y

  def score(self, X, y):
    y_pred = self.predict(X)
    y_same = np.zeros(y_pred.shape[0])
    y_same[y_pred == y] = 1
    return np.mean(y_same)

class ClassifierWithPerfDist:
  def __init__(self, classifier=SVC, percentile=95, n_stdevs=None, **kwargs): 
    dflg_classif = {}
    dflg_classif.update(kwargs)
    self.classifier = classifier(**dflg_classif)
    self.percentile = percentile 
    self.n_stdevs   = n_stdevs
    self.latt_to_cut = { latt.name : latt.outlier_cut for latt in cnst.lattices }

  def fit(self, X, y):
    self.classifier.fit(X, y)
    if self.percentile == None:
      return self
    
    for latt in tqdm(cnst.lattices):
      perfx = perf_features[latt.name]
      lattX = X[y==latt.y_label][:]
      latt_dist = np.linalg.norm(lattX - perfx, axis=-1) \
                  * cosine_distances(lattX, np.expand_dims(perfx, axis=0))
      if self.percentile != None:
        self.latt_to_cut[latt.name] = np.percentile(latt_dist, self.percentile)
      elif self.n_stdevs != None:
        mean = np.mean(latt_dist)
        std = np.std(latt_dist)
        self.latt_to_cut[latt.name] = mean + self.n_stdevs * std
      else:
        raise ValueError('either self.percentile or n_stdevs must have numeric value')

    print('done training')
    return self

  def get_params(self, *kargs, **kwargs):
    params = self.classifier.get_params(*kargs, **kwargs)
    params['percentile'] = self.percentile
    params['n_stdevs']   = self.n_stdevs
    return params

  def set_params(self, *kargs, percentile=95, n_stdevs=None, **kwargs):
    params['percentile'] = percentile
    params['n_stdevs']   = n_stdevs
    return self.classifier.set_params(*kargs, **kwargs)

  def predict(self, X):
    y = self.classifier.predict(X)
    for latt in tqdm(cnst.lattices):
      perf = perf_features[latt.name]
      y_latt = y[y==latt.y_label]
      if not len(y_latt):
        continue
      X_latt = X[y==latt.y_label][:]
      cutoff = self.latt_to_cut[latt.name]
      dist = np.linalg.norm(X_latt - perf, axis=-1) \
          * cosine_distances(X_latt, np.expand_dims(perf, axis=0))[:,0]
      y_latt[dist > cutoff] = -1
      y[y==latt.y_label] = y_latt
    return y

  def score(self, X, y):
    y_pred = self.predict(X)
    y_same = np.zeros(y_pred.shape[0])
    y_same[y_pred == y] = 1
    return np.mean(y_same)
