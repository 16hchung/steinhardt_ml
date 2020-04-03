from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np

from util import constants as cnst
n_latt = len(cnst.lattices)

# implemented same functions listed for sklearn SVM classifier classes
# note: # classes is predetermined
class MultiOutlierClassifier: #NOTE: Classes are 1 indexed
  def __init__(self, n_classes=n_latt, single_model=OneClassSVM, **kwargs):
    self.n_classes = n_classes
    self.single_model = single_model
    self.cls_to_svm = [single_model(**kwargs) for _ in range(n_classes)]
    self.df_scalers = [StandardScaler(with_mean=0, with_std=1) for _ in range(n_classes)]


  def fit(self, X, y):
    if y[(y<=0) | (y>self.n_classes)].size > 0:
      raise ValueError('y contains values outside range from 1 to '+str(self.n_classes))

    for cls, svm in enumerate(self.cls_to_svm):
      specificX = X[y==cls+1]
      svm.fit(specificX)

    for cls, scaler in enumerate(self.df_scalers):
      specificX = X[y==cls+1]
      df = self.decision_function(specificX)[:,cls]
      df = df[df>0] # get rid of outliers before scaling
      scaler.fit(df[:,np.newaxis])

    return self


  def get_params(self, *kargs, **kwargs):
    return self.cls_to_svm[0].get_params(*kargs, **kwargs)
    #return {cls:svm.get_params(*kargs, **kwargs) for cls, svm in enumerate(self.cls_to_svm)}
    #return [svm.get_params(*kargs, **kwargs) for svm in self.cls_to_svm]


  def set_params(self, *kargs, **kwargs):
    for svm in self.cls_to_svm:
      svm.set_params(*kargs, **kwargs)
    return self


  def predict(self, X):
    scores = self.decision_function(X, scaled=True)
    y_pred = np.argmax(scores, axis=1) + 1
    max_scores = np.max(scores, axis=1)
    y_pred[max_scores < 0] = -1
    return y_pred


  def score(self, X, y):
    y_pred = self.predict(X)
    y_same = np.zeros(y_pred.shape[0])
    y_same[y_pred == y] = 1
    return np.mean(y_same)


  def decision_function(self, X, scaled=False):
    scores = []
    for svm in self.cls_to_svm:
      scores.append(svm.decision_function(X)[:, np.newaxis])
    df = np.concatenate(scores, axis=1)
    if scaled:
      is_outlier = df < 0
      for cls, scaler in enumerate(self.df_scalers):
        df[:, cls] = scaler.transform(df[:,cls,np.newaxis])[:,0]
      df[is_outlier] = -1
    return df

'''
TODO
- concat all sets of features (ie x = [x_12; x_8]) -- in class that subclasses this
'''
