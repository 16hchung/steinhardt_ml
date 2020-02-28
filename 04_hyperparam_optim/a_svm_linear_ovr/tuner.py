from sklearn.svm import LinearSVC

from ..z_helpers import a_grid_search as gs
from ..z_helpers import b_learning_curve as lc
from ..z_helpers import c_model_scores as ms
from ..z_helpers.model_tuner import ModelTuner
from util import constants as cnst
from util import dir_util


class ModelTunerA(ModelTuner):
  def __init__(self):
    model = LinearSVC
    model_args = {'class_weight':'balanced'}
    super().__init__(model, model_args, cnst.svm_lin_ovr_path)
    self.hyperprm_sffx = ''

  def set_hyperparam(self):
    self.model_params['C'] = .3
    self.hyperprm_sffx = '_C_3e-1'

if __name__=='__main__':
  tuner = ModelTunerA()
  tuner.cmdline_main()
