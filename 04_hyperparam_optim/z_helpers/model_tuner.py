''' Abstract class for optimizing specific models '''
import numpy as np

from util import dir_util

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
    elif stage == 'ms2':
      self.ms_plot()
    elif stage == 'ms':
      self.ms_compute()
      self.ms_plot()
    else:
      print('invalid stage argument')

  # Load data set.
  def load_data(self):
    paths = dir_util.clean_features_paths02()
    X = np.loadtxt(paths.X)
    y = np.loadtxt(paths.y)
    return X,y

  ######## Grid search functions ###############

  def gs_compute(self):
    pass

  def gs_plot(self):
    pass

  def lc_compute(self):
    pass

  def lc_plot(self):
    pass

  def ms_compute(self):
    pass

  def ms_plot(self):
    pass
