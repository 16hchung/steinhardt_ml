import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from tqdm import tqdm
import joblib

from util import dir_util
from util import constants as cnst
from util import calc

################################################################################
# Train model.                                                                 #
################################################################################
def run(X,y, model, model_params, model_path, scores_path, relbl_wrong_neigh=False, eval_liq=False):
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

def run_concated(X, y, model, model_params, model_path, scores_path, eval_liq=False):
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

def run_all_concated(X, y, model, model_params, model_path, scores_path, eval_liq=False, baseline=False):
  params = {'tol':1e-3,'max_iter':1000, 'verbose':True}
  params.update(model_params)
  
  X,y = shuffle(X,y)
  n_train = 50000
  X = X[:n_train]
  y = y[:n_train]
  
  #X = X[:,150:] # TODO
  # Fit on train set.
  clf = model(**params)
  #clf = model(**model_params)
  clf.fit(X,y)
  joblib.dump(clf,model_path)

  n_test = 10000
  paths = dir_util.clean_features_paths02(istest=True)
  X_valid = np.loadtxt(paths.X.format('concat_'))
  #X_valid = np.loadtxt(paths.X.format('concat_'))[:,150:] # TODO
  y_valid = np.loadtxt(paths.y.format('concat_'))
  # include liquid points
  if eval_liq:
    paths = dir_util.clean_features_paths02(istest=True, liq=True)
    X_liq = np.loadtxt(paths.X.format('concat_'))
    y_liq = np.full(len(X_liq), -1)
    X_valid = np.row_stack([X_valid, X_liq])
    y_valid = np.concatenate([y_valid, y_liq])

  # limit number of test points for overall accuracy
  X_valid, y_valid = shuffle(X_valid, y_valid)
  X_valid = X_valid[:n_test*len(cnst.lattices)]
  y_valid = y_valid[:n_test*len(cnst.lattices)]
  save_scores(clf, X_valid, y_valid, scores_path.format('cat_'))

  #if not eval_liq:
  if True:
    ml_key = 'ML_baseline' if baseline else 'ML'
    scores = {'latt': [], 'temp': [], ml_key: []}
    for latt in tqdm(cnst.lattices):
      for temp in tqdm(range(latt.low_temp, latt.high_temp+latt.step_temp, latt.step_temp)):
        paths = dir_util.clean_features_paths02(istest=True, lattice=latt, temp=temp)
        X_valid = shuffle(np.loadtxt(paths.X.format('concat_')))[:n_test, :]
        #X_valid = shuffle(np.loadtxt(paths.X.format('concat_')))[:n_test, 150:]
        y_valid = np.ones(X_valid.shape[0]) * latt.y_label
        scores['latt'].append(latt.name)
        scores['temp'].append(temp)
        scores[ml_key].append(clf.score(X_valid, y_valid))
    df = pd.DataFrame(data=scores)
    df.to_csv(scores_path.format('cat_byT_'), index=False)


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
    
