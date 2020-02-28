import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib

from util import dir_util
from util import constants as cnst

################################################################################
# Train model.                                                                 #
################################################################################
def run(X,y, model, model_params, model_path, scores_path, relbl_wrong_neigh=False):
  params = {'tol':1e-3,'max_iter':1000}
  params.update(model_params)
  
  # Fit on train set.
  clf = model(**params)
  clf.fit(X,y)
  joblib.dump(clf,model_path)

  paths = dir_util.clean_features_paths02(istest=True)
  for n_neigh in cnst.possible_n_neigh:
    X_valid = np.loadtxt(paths.X.format(n_neigh))
    y_valid = np.loadtxt(paths.y.format(n_neigh))
    if relbl_wrong_neigh:
      y_valid = relabel_wrong_neigh(y_valid, n_neigh)
    save_scores(clf, X_valid, y_valid, scores_path.format(n_neigh))

################################################################################
# Compute validation set accuracy and confusion matrix.                        #
################################################################################
def save_scores(clf, X_valid, y_valid, scores_path):
  with open(scores_path, 'w') as f:
    acc_valid = clf.score(X_valid,y_valid) # Accuracy on the validation set.
    f.write('Accuracy on validation set: %.2f\n\n' % (acc_valid*100))
    f.write('%s\n' % classification_report(y_valid,clf.predict(X_valid)))
    f.write('Confusion Matrix:\n')
    f.write('%s' % confusion_matrix(y_valid,clf.predict(X_valid)))


def get_test_data():
  paths = dir_util.clean_features_paths02(istest=True)
  neigh_to_X = {n : np.loadtxt(paths.X.format(n)) for n in cnst.possible_n_neigh}
  neigh_to_y = {n : np.loadtxt(paths.y.format(n)) for n in cnst.possible_n_neigh}
  X = np.array([[]*cnst.n_features])
  y = np.array([])

  for latt in cnst.lattices:
    neighX = neigh_to_X[latt.n_neigh]
    neighy = neigh_to_y[latt.n_neigh]
    
def relabel_wrong_neigh(y_valid, n_neigh):
  for latt in cnst.lattices:
    if latt.n_neigh == n_neigh:
      continue
    y_valid[y_valid == latt.y_label] = -1
  return y_valid
