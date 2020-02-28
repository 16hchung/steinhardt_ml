''' Abstract class for optimizing specific models '''
import numpy as np

from util import dir_util
from . import a_grid_search as gs
from . import b_learning_curve as lc
from . import c_model_scores as ms

class ModelTuner:
  def __init__(self, model, model_params, model_dir):
    self.model = model
    self.model_params = model_params
    # properties keeping track of fnames/paths
    self.model_dir = model_dir
    self.all_paths = dir_util.hyperparam_all_paths04(model_dir)
    self.gs_paths = self.all_paths.grid_search
    self.lc_paths = self.all_paths.learning_curve
    self.ms_paths = self.all_paths.model_score
    # this class method options
    self.should_relbl_wrong_neigh = False

  def cmdline_main(self):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--stage', type=str, help='options: gs<1/2> (grid search), lc<1/2> (learning curve), ms<1/2> (model score)')
    args = parser.parse_args()
    stage = args.stage

    if stage == 'gs1':
      self.gs_compute()
    elif stage == 'gs2':
      self.gs_plot()
    elif stage == 'gs':
      self.gs_compute()
      self.gs_plot()
    elif stage == 'lc1':
      self.lc_compute()
    elif stage == 'lc2':
      self.lc_plot()
    elif stage == 'lc':
      self.lc_compute()
      self.lc_plot()
    elif stage == 'ms1':
      self.ms_compute()
    else:
      print('invalid stage argument')

  # Load data set.
  def load_data(self):
    paths = dir_util.clean_features_paths02(pseudo=True)
    X = np.loadtxt(paths.X.format('adapt_'))
    y = np.loadtxt(paths.y.format('adapt_'))
    return X,y

  def set_hyperparam(self):
    pass

  ######## Grid search functions ###############

  def gs_compute(self):
    X,y = self.load_data()
    gs.compute.grid_search_C_only(self.model, self.model_params, X, y, self.gs_paths)

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
    ms.compute.run(X,y, self.model, self.model_params, self.ms_paths.model_tmplt.format(hyperprm_sffx=self.hyperprm_sffx), self.ms_paths.scores, relbl_wrong_neigh=self.should_relbl_wrong_neigh)

