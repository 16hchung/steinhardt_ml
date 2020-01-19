import numpy as np
from sklearn.model_selection import learning_curve

from util import dir_util

################################################################################
# Compute.                                                                     #
################################################################################

def run(X,y, model, model_params, save_path):
  params = {'tol':1e-3,'max_iter':1000}
  params.update(model_params)

  # Compute learning curve.
  m, acc_train, acc_valid = learning_curve(model(**params), X, y, train_sizes=np.linspace(0.01,1.0,30), cv=5, n_jobs=-2)

  # Compute mean and errors.
  acc_train_avg = np.mean(acc_train, axis=1)
  acc_train_std = np.std(acc_train, axis=1)
  acc_valid_avg = np.mean(acc_valid, axis=1)
  acc_valid_std = np.std(acc_valid, axis=1)
  data = np.array([m, acc_train_avg, acc_train_std, acc_valid_avg, acc_valid_std])

  # Save results.
  np.savetxt(save_path, data.T, header=' train set size |       training score | valid score', fmt='%5d %.5f %.5f %.5f %.5f')

################################################################################
