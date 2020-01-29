from . import constants as cnst
from collections import namedtuple
from pathlib import Path

blank = '{}'
pseudo_pre = 'pseudo_'

def dump_path_for_lattice00(latt, perfect=False):
  perf_suffix = '_perfect' if perfect else ''
  dump_tmpl = 'data/dump/dump_{}{}_{}.dat'
  return make_dirs(cnst.md_path + latt.sim_dir + dump_tmpl.format(latt.name, perf_suffix, blank))[0]

def all_features_path01(latt, pseudo=False):
  pseudo_prefix = pseudo_pre if pseudo else ''
  return make_dirs('{}data/X/{}X_{}.dat'.format(cnst.raw_feat_path, pseudo_prefix, latt.name))[0]

def synth_carteasian_path01(latt):
  return make_dirs('{}data/synth_cart/{}.dat'.format(cnst.raw_feat_path, latt.name))[0]

def clean_features_paths02(istest=False, pseudo=False):
  pseudo_prefix = pseudo_pre if pseudo else ''
  split_lbl = 'test' if istest else 'train'
  tmplt = '{par_dir}data/{ps}{blank}{blank}_{split_lbl}.dat'.format(
    ps=pseudo_prefix, par_dir=cnst.clean_feat_path, blank=blank, split_lbl=split_lbl
  )

  unscaledX = tmplt.format('X', '_unscaled')
  X = tmplt.format('X', '')
  y = tmplt.format('y', '')

  Paths = namedtuple('Paths', 'unscaledX X y')
  return Paths(*make_dirs(unscaledX, X, y))

def scaler_path02(pseudo=False):
  pseudo_prefix = pseudo_pre if pseudo else ''
  return '{}data/{}scaler.pkl'.format(cnst.clean_feat_path, pseudo_prefix)

def pca_data_paths03(pseudo=False):
  pseudo_prefix = pseudo_pre if pseudo else ''
  tmplt = '{}data/{}{}{}.dat'.format(cnst.pca_path, pseudo_prefix, blank, blank)
  comp_prefix = 'PCA_component_'
  comp1 = tmplt.format(comp_prefix, 1)
  comp2 = tmplt.format(comp_prefix, 2)
  variance = tmplt.format('variance', '')
  
  Paths = namedtuple('Paths', 'comp1 comp2 variance')
  return Paths(*make_dirs(comp1, comp2, variance))

def pca_fig_path03():
  return '{}fig_PCA.png'.format(cnst.vis_figures_path)

def tSNE_data_paths03(pseudo=False):
  pseudo_prefix = pseudo_pre if pseudo else ''
  tmplt = '{}data/{}{}.data'.format(cnst.tSNE_path, pseudo_prefix, blank)
  y = tmplt.format('y')
  X = tmplt.format('tSNE_{}')
  X_with_PCA = tmplt.format('tSNE_PCA_{}')
  Paths = namedtuple('Paths', 'y X_tmplt X_with_PCA_tmplt')
  return Paths(*make_dirs(y, X, X_with_PCA))

def tSNE_fig_tmplts03():
  tmplt = '{}fig_tSNE{}_{}.png'.format(cnst.vis_figures_path, blank, blank)
  no_PCA = tmplt.format('', blank)
  with_PCA = tmplt.format('_PCA', blank)
  Paths = namedtuple('Paths', 'no_PCA with_PCA')
  return Paths(*make_dirs(no_PCA, with_PCA))

def feat_pairs_map_path03():
  return make_dirs('{}fig_Q_vs_Q.png'.format(cnst.vis_figures_path))[0]

def grid_search_paths04(model_dir, tmplt):
  data_path = 'grid_search_data/'
  best_params = tmplt.format(subdir=data_path, fname='best_parameters.dat')
  val_curve_data = tmplt.format(subdir=data_path, fname='validation_curve_C{}.dat')
  fig_path = 'grid_search_figures/'
  val_curve_fig = tmplt.format(subdir=fig_path, fname='validation_curve_C{}.png')
  Paths = namedtuple('Paths', 'best_params_data val_curve_data_tmplt val_curve_fig_tmplt')
  return Paths(*make_dirs(best_params, val_curve_data, val_curve_fig))

def learning_curve_paths04(model_dir, tmplt):
  subdir = 'learn_curve/'
  data_path = tmplt.format(subdir=subdir, fname='data.dat')
  fig_path  = tmplt.format(subdir=subdir, fname='fig.png')
  Paths = namedtuple('Paths', 'data fig')
  return Paths(*make_dirs(data_path, fig_path))

def model_score_paths04(model_dir, tmplt):
  subdir = 'model_score/'
  model = tmplt.format(subdir=subdir, fname='model{hyperprm_sffx}.pkl')
  scores = tmplt.format(subdir=subdir, fname='scores.dat')
  Paths = namedtuple('Paths', 'model_tmplt scores')
  return Paths(*make_dirs(model, scores))

def hyperparam_all_paths04(model_dir):
  tmplt = '{}{}{}{}'.format(cnst.hyperparam_optim_path, model_dir, '{subdir}', '{fname}')
  AllPaths = namedtuple('AllPaths', 'grid_search learning_curve model_score')
  grid_search = grid_search_paths04(model_dir      , tmplt)
  learning_curve = learning_curve_paths04(model_dir, tmplt)
  model_score = model_score_paths04(model_dir      , tmplt)
  return AllPaths(grid_search, learning_curve, model_score)




##################### HELPERS ##########################

def make_dirs(*paths):
  for p in paths:
    Path(p).parents[0].mkdir(parents=True, exist_ok=True)
  return paths

