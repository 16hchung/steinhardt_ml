import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

from util import constants as cnst, dir_util

if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--model',type=str,default='f')
  parser.add_argument('--baseline',action='store_true')
  args = parser.parse_args()
  
  model_path  = cnst.svm_lin_ovr_path     if 'a' == args.model else \
                cnst.svm_lin_ovo_path     if 'b' == args.model else \
                cnst.svm_rbf_ovo_path     if 'c' == args.model else \
                cnst.ocsvm_rbf_path       if 'd' == args.model else \
                cnst.all_svm_lin_ovo_path if 'e' == args.model else \
                cnst.cat_svm_lin_ovr_path if 'f' == args.model else \
                cnst.cat_svm_rbf_ovo_path if 'g' == args.model else \
                cnst.cat_svm_lin_ovo_path if 'h' == args.model else \
                cnst.cat_with_liq_perf_path if 'j' == args.model else \
                ''
  fnames = dir_util.model_exam_paths06(model_path, baseline=args.baseline)
  hyperparam_fnames = dir_util.hyperparam_all_paths04(model_path)

  cm = plt.get_cmap('gist_rainbow')

  other_scores = pd.read_csv(fnames.other_scores,na_values='None',skipinitialspace=True)
  model_scores = pd.read_csv(hyperparam_fnames.model_score.scores.format('cat_byT_'))
  df = pd.merge(other_scores, model_scores, on=['latt','temp'], how='inner')
  if args.baseline:
    baseline_fnames = dir_util.hyperparam_all_paths04(model_path, baseline=True)
    baseline_scores = pd.read_csv(baseline_fnames.model_score.scores.format('cat_byT_'))
    df = pd.merge(df, baseline_scores, on=['latt','temp'], how='inner')
    
  df.to_csv(fnames.model_scores,na_rep='None')
  
  # normalize temperature
  for latt in cnst.lattices:
    df.loc[df.latt==latt.name, 'temp'] /= float(latt.T_m)

  # TODO set flag
  df = df.rename(columns={
    'PTM_dflt': r'$PTM_{dflt}$',
    'PTM': r'$PTM_{tuned}$',
    'ML': 'LattSVM'
  })
  df = df[['latt', 'temp', r'$PTM_{dflt}$', r'$PTM_{tuned}$', 'LattSVM']]

  # plot for each lattice
  df.set_index('temp', inplace=True)

  for latt in cnst.lattices:
    plt.rcParams.update({'font.size': 16, 'figure.autolayout': True})
    df.groupby('latt').get_group(latt.name).plot(legend=True)
    plt.axvline(x=1,ls='--', c='k', lw=1.0)
    plt.title(f'Test Accuracy for {latt.name.upper()}')
    plt.xlabel(r'$T/T_m$')
    plt.ylabel('Accuracy')
    #plt.tight_layout()
    plt.savefig(fnames.fig_tmplt.format(latt.name), dpi=300)
    plt.clf()

