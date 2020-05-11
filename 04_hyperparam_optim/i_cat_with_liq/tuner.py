import numpy as np

from full_model.outlier_pipe import ClassifierWithLiq
from ..z_helpers import c_model_scores as ms
from ..z_helpers.model_tuner import ModelTuner
from util import constants as cnst, dir_util

class ModelTunerI(ModelTuner):
  def __init__(self):
    model = ClassifierWithLiq
    model_args = {
        'outlier_args': {'tol':1e-3, 'max_iter':10000, 'nu':.15},
        'class_args': {
          'tol':1e-3, 'max_iter':100000, 'cache_size':1000, 'class_weight':'balanced',
          'C': 10, 'gamma': .03
        }
    }
    super().__init__(ClassifierWithLiq, model_args, cnst.cat_with_liq_path)
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

if __name__=='__main__':
  tuner = ModelTunerI()
  tuner.cmdline_main()
