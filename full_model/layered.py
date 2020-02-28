import numpy as np

from .outlier_detect import MultiOutlierClassifier

class LayeredMultiOutlierClassifier(MultiOutlierClassifier):
  def __init__(self, n_feat_sets, cls_to_feat_set, *kargs, **kwargs):
    super().__init__(*kargs, **kwargs)
    self.n_feat_sets = n_feat_sets # eg number of possible n_neighbors
    self.cls_to_feat_set = cls_to_feat_set # map {label : index of feature set}

  def fit(self, X, y):
    pass

  def predict(self, X, y):
    pass

  def decision_function(self, X, y):
    pass

  def score(self, X, y):
    # mark correct prediction of cls gets hit for correct feature set
    # or if y_pred = -1 for wrong feature set
    pass

  # rest covered in super class


