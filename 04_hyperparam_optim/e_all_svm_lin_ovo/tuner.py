import numpy as np

from full_model.all_data_trained import AllDataLinearSVC, AllDataLinearSVCLayered
from ..z_helpers import a_grid_search as gs
from ..z_helpers import b_learning_curve as lc
from ..z_helpers import c_model_scores as ms
from ..z_helpers.model_tuner import ModelTuner
from util import constants as cnst
from util import dir_util

class ModelTunerE(ModelTuner):
  def __init__(self, layered):
    model = AllDataLinearSVCLayered if layered else AllDataLinearSVC
    model_args = {'max_iter':5000}
    #model_args = {'nu': .2}
    super().__init__(model, model_args, cnst.all_svm_lin_ovo_path)
    self.hyperprm_sffx = ''
    self.should_relbl_wrong_neigh = True
    self.concat_test = layered

  def load_data(self):
    paths = dir_util.clean_features_paths02(pseudo=True)
    Xs = []
    ys = []
    for i_neigh, neigh in enumerate(cnst.possible_n_neigh):
      X = np.loadtxt(paths.X.format(neigh))
      y = np.loadtxt(paths.y.format(neigh))
      Xs.append(X)
      ys.append(y)
    X = np.row_stack(Xs)
    y = np.concatenate(ys)
    return X,y

  def set_hyperparam(self):
    pass


if __name__=='__main__':
  tuner = ModelTunerE(True)
  tuner.cmdline_main()
