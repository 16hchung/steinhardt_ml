import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import joblib

from util import dir_util
from util import constants as cnst
from util import calc

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
      y_valid = calc.relabel_wrong_neigh(y_valid, n_neigh)
    save_scores(clf, X_valid, y_valid, scores_path.format(n_neigh))

def run_concated(X, y, model, model_params, model_path, scores_path):
  params = {'tol':1e-3,'max_iter':1000}
  params.update(model_params)
  
  # Fit on train set.
  clf = model(**params)
  clf.fit(X,y)
  joblib.dump(clf,model_path)

  paths = dir_util.clean_features_paths02(istest=True)
  y_valid = np.absolute(np.loadtxt(paths.y.format(cnst.possible_n_neigh[0]))) # abs of all ys are the same
  Xs_valid = []
  for n_neigh in cnst.possible_n_neigh:
    X_valid = np.loadtxt(paths.X.format(n_neigh))
    Xs_valid.append(X_valid)
  X_valid = np.concatenate(Xs_valid, axis=1)
  save_scores(clf, X_valid, y_valid, scores_path.format('all'))

def run_all_concated(X, y, model, model_params, model_path, scores_path):
  params = {'tol':1e-3,'max_iter':1000}
  params.update(model_params)
  
  # Fit on train set.
  clf = model(**params)
  clf.fit(X,y)
  joblib.dump(clf,model_path)

  n_test = 10000
  paths = dir_util.clean_features_paths02(istest=True)
  X_valid = np.loadtxt(paths.X.format('concat_'))
  y_valid = np.loadtxt(paths.y.format('concat_'))
  # limit number of test points for overall accuracy
  X_valid, y_valid = shuffle(X_valid, y_valid)
  X_valid = X_valid[:n_test*len(cnst.lattices)]
  y_valid = y_valid[:n_test*len(cnst.lattices)]
  save_scores(clf, X_valid, y_valid, scores_path.format('cat_'))

  scores = {'latt': [], 'temp': [], 'accuracy': []}
  for latt in cnst.lattices:
    for temp in range(latt.low_temp, latt.high_temp+latt.step_temp, latt.step_temp):
      paths = dir_util.clean_features_paths02(istest=True, lattice=latt, temp=temp)
      X_valid = shuffle(np.loadtxt(paths.X.format('concat_')))[:n_test]
      y_valid = np.ones(X_valid.shape[0]) * latt.y_label
      scores['latt'].append(latt.name)
      scores['temp'].append(temp)
      scores['accuracy'].append(clf.score(X_valid, y_valid))
  df = pd.DataFrame(data=scores)
  df.to_csv(scores_path.format('cat_byT_'))


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
    
