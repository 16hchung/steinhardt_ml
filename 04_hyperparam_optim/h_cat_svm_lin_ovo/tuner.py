from sklearn.svm import SVC
import numpy as np

from ..z_helpers import a_grid_search as gs
from ..z_helpers import b_learning_curve as lc
from ..z_helpers import c_model_scores as ms
from ..z_helpers.model_tuner import ModelTuner
from util import constants as cnst
from util import dir_util

class ModelTunerH(ModelTuner):
  def __init__(self):
    model = SVC
    model_args = {'max_iter':10000,'kernel':'linear', 'cache_size':1000, 'class_weight':'balanced'}
    super().__init__(model, model_args, cnst.cat_svm_lin_ovo_path)
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

  def set_hyperparam(self):
    self.model_params['C'] = .03
    self.hyperprm_sffx = '_C_3e-1'

if __name__=='__main__':
  tuner = ModelTunerH()
  tuner.cmdline_main()
