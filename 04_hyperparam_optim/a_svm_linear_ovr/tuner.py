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

  def set_hyperparam(self):
    self.model_params['C'] = .3

  def gs_compute(self):
    X,y = self.load_data()
    gs.compute.grid_search_C_only(self.model, self.model_args, X, y, self.gs_paths)

  def gs_plot(self):
    gs.plot.plot_validation('C', self.gs_paths.val_curve_data_tmplt.format(''), self.gs_paths.val_curve_fig_tmplt.format(''))

  def lc_compute(self):
    X,y = self.load_data()
    self.set_hyperparam()
    lc.compute.run(X, y, self.model, self.model_params, self.lc_paths.data)

  def lc_plot(self):
    data_path = self.lc_paths.data
    fig_path = self.lc_paths.fig
    lc.plot.plot_learning(data_path, fig_path)

  def ms_compute(self):
    X,y = self.load_data()
    self.set_hyperparam()
    ms.compute.run(X,y, self.model, self.model_params, self.ms_paths.model_tmplt.format(hyperprm_sffx='_C_3e-1'), self.ms_paths.scores)

if __name__=='__main__':
  tuner = ModelTunerA()
  tuner.cmdline_main()
