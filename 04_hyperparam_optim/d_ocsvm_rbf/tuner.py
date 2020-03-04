from full_model.outlier_detect import MultiOutlierClassifier
from ..z_helpers import a_grid_search as gs
from ..z_helpers import b_learning_curve as lc
from ..z_helpers import c_model_scores as ms
from ..z_helpers.model_tuner import ModelTuner
from util import constants as cnst
from util import dir_util

class ModelTunerD(ModelTuner):
  def __init__(self):
    model = MultiOutlierClassifier
    model_args = {'max_iter':10000}
    #model_args = {'nu': .2}
    super().__init__(model, model_args, cnst.ocsvm_rbf_path)
    self.hyperprm_sffx = ''
    self.should_relbl_wrong_neigh = True

  def set_hyperparam(self):
    self.model_params['nu'] = .15

if __name__=='__main__':
  tuner = ModelTunerD()
  tuner.cmdline_main()
