from sklearn.svm import SVC, OneClassSVM
import numpy as np


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

