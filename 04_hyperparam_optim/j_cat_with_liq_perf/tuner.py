import numpy as np

from full_model.outlier_pipe import ClassifierWithPerfDist
from ..z_helpers import c_model_scores as ms
from ..z_helpers.model_tuner import ModelTuner
from util import constants as cnst, dir_util

class ModelTunerJ(ModelTuner):
  def __init__(self):
    model_args = {
        'tol':1e-3, 'max_iter':100000, 'cache_size':1000, 'class_weight':'balanced',
        'C': 10, 'gamma': .03
    }
    super().__init__(ClassifierWithPerfDist, model_args, cnst.cat_with_liq_perf_path)
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
    self.model_params['C'] = 10
    self.model_params['gamma'] = .01
    self.model_params['cutoff_scaler'] = 1
    self.model_params['percentile'] = 95
    self.hyperprm_sffx = '_correct_scaler_C10_g.01_integ95_euccos'

if __name__=='__main__':
  tuner = ModelTunerJ()
  tuner.cmdline_main()
