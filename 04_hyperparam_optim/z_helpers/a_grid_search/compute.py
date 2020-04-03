import numpy as np
from sklearn.model_selection import GridSearchCV
from copy import copy

from util import dir_util, constants as cnst

param_vals = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 1e1, 3e1, 1e2, 3e2, 1e3]
prob_param_vals = [.1*i for i in range(1,10)]

################################################################################
# Input parameters and setup.                                                  #
################################################################################

# Build parameter grid.
# TODO maybe 2D grid with Gamma?
def get_C_param_grid():
  C = copy(param_vals)
  param_grid = [{'C': np.array(C)}]
  return C, param_grid

def get_nu_param_grid():
  nu = copy(param_vals)
  param_grid = [{'nu': np.array(nu)}]
  return nu, param_grid

################################################################################
# Compute.                                                                     #
################################################################################

def grid_search_C(model, X, y, param_grid, val_curve_path, c_name='C', best_params_path=None):
  # Perform grid search.
  clf = GridSearchCV(model, param_grid, cv=5, n_jobs=-2, iid=False, return_train_score=True, verbose=2)
  clf.fit(X,y)
  C_arr = param_grid[0][c_name]

  # Save best parameters.
  if best_params_path != None:
    with open(best_params_path, 'w') as f:
      f.write('Best parameters: %r\n' % clf.best_params_)
      f.write('Accuracy: %.5f' % clf.best_score_)

  # Construct validation curves for C: linear kernel.
  N = len(C_arr)
  data = np.zeros((5,N))
  data[0,:] = C_arr
  data[1,:] = np.array(clf.cv_results_['mean_test_score'])[:N]
  data[2,:] = np.array(clf.cv_results_['std_test_score'])[:N]
  data[3,:] = np.array(clf.cv_results_['mean_train_score'])[:N]
  data[4,:] = np.array(clf.cv_results_['std_train_score'])[:N]
  np.savetxt(val_curve_path, data.T, header=' '+c_name+' | training score | valid score', fmt='%.0e %.5f %.5f %.5f %.5f')

dflt_model_args = {'max_iter':1000, 'tol':1e-3}

def grid_search_C_only(model, model_args, X, y, gs_paths):
  C, param_grid = get_C_param_grid()
  model_args.update(dflt_model_args) # merge arguments
  grid_search_C(model(**model_args), X, y, param_grid, gs_paths.val_curve_data_tmplt.format(''), best_params_path=gs_paths.best_params_data)

def grid_search_C_and_gamma(model, model_args, X, y, gs_paths, get_param_grid=None, c_name='C'):
  if get_param_grid == None:
    C, param_grid = get_C_param_grid()
  else:
    C, param_grid = get_param_grid()
  gamma = copy(param_vals)
  model_args.update(dflt_model_args)
  for igamma, gamma_val in enumerate(gamma):
    model_args['gamma'] = gamma_val
    gamma_suffix = '_{}'.format(igamma+1)
    val_fname = gs_paths.val_curve_data_tmplt.format(gamma_suffix)
    grid_search_C(model(**model_args), X, y, param_grid, val_fname, c_name=c_name)

################################################################################
