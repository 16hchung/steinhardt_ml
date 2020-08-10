import numpy as np
from sklearn.svm import SVC

from ..z_helpers import a_grid_search as gs
from ..z_helpers import b_learning_curve as lc
from ..z_helpers import c_model_scores as ms
from ..z_helpers.model_tuner import ModelTuner
from util import constants as cnst
from util import dir_util

class ModelTunerG(ModelTuner):
  def __init__(self):
    model_args = {'max_iter':100000, 'cache_size':1000, 'class_weight':'balanced'}
    #model_args = {'nu': .2}
    super().__init__(SVC, model_args, cnst.cat_svm_rbf_ovo_path)
    self.hyperprm_sffx = ''
    self.train_concated = True

  def load_data(self):
    if self.baseline: 
      paths = dir_util.clean_features_paths02(istest=True)
    else:
      paths = dir_util.clean_features_paths02(pseudo=True)
    X = np.loadtxt(paths.X.format('concat_'))
    y = np.loadtxt(paths.y.format('concat_'))
    return X,y

  def gs_compute(self):
    X,y = self.load_data()
    gs.compute.grid_search_C_and_gamma(self.model, self.model_params, X, y, self.gs_paths)

  def gs_plot(self):
    gs.plot.plot_grid(self.gs_paths)

  def set_hyperparam(self):
    #self.model_params['C'] = 10
    #self.model_params['gamma'] = .01
    self.hyperprm_sffx = '_dflt'


if __name__=='__main__':
  tuner = ModelTunerG()
  tuner.cmdline_main()
