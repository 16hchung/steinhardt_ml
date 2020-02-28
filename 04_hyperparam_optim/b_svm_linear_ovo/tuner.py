from sklearn.svm import SVC

from ..z_helpers import a_grid_search as gs
from ..z_helpers import b_learning_curve as lc
from ..z_helpers import c_model_scores as ms
from ..z_helpers.model_tuner import ModelTuner
from util import constants as cnst
from util import dir_util

class ModelTunerB(ModelTuner):
  def __init__(self):
    model = SVC
    model_args = {'kernel':'linear', 'cache_size':1000}
    super().__init__(model, model_args, cnst.svm_lin_ovo_path)
    self.hyperprm_sffx = ''

  def set_hyperparam(self):
    self.model_params['C'] = .03
    self.hyperprm_sffx = '_C_3e-1'

if __name__=='__main__':
  tuner = ModelTunerB()
  tuner.cmdline_main()
