import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

from util import constants as cnst, dir_util

if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--model',type=str,default='f')
  args = parser.parse_args()
  
  model_path  = cnst.svm_lin_ovr_path     if 'a' == args.model else \
                cnst.svm_lin_ovo_path     if 'b' == args.model else \
                cnst.svm_rbf_ovo_path     if 'c' == args.model else \
                cnst.ocsvm_rbf_path       if 'd' == args.model else \
                cnst.all_svm_lin_ovo_path if 'e' == args.model else \
                cnst.cat_svm_lin_ovo_path if 'f' == args.model else \
                ''
  fnames = dir_util.model_exam_paths06(model_path)
  hyperparam_fnames = dir_util.hyperparam_all_paths04(model_path)

  cm = plt.get_cmap('gist_rainbow')

  methods = [['PTM', 'CNA', 'AJA', 'VTM'],
             ['PTM', 'CNA', 'AJA', 'VTM'],
             ['PTM', 'CNA', 'AJA', 'VTM'],
             ['PTM', 'CNA', 'CPA'],
             ['PTM', 'CNA', 'CPA'],
             ['PTM']]

  method_to_color = {'PTM':'C0', 'CNA':'C1', 'AJA':'C2', 'VTM':'C3', 'CPA':'C4', 'ML':'Machine Learning'}
  method_to_name = {'PTM':'Polyhedral Template Matching', 'CNA':'Common Neighbor Analysis', 'AJA':'Ackland-Jones Analysis', 'VTM':'VoroTop Analysis', 'CPA':'Chill+'}

  other_scores = pd.read_csv(fnames.other_scores,na_values='None',skipinitialspace=True)
  model_scores = pd.read_csv(hyperparam_fnames.model_score.scores.format('cat_byT_'))
  df = pd.merge(other_scores, model_scores, on=['latt','temp'], how='outer')
  df.to_csv(fnames.model_scores,na_rep='None')
  
  # normalize temperature
  for latt in cnst.lattices:
    df.loc[df.latt==latt.name, 'temp'] /= float(latt.T_m)

  # plot for each lattice
  df.set_index('temp', inplace=True)
  for latt in cnst.lattices:
    df.groupby('latt').get_group(latt.name).plot(legend=True)
    plt.axvline(x=1,ls='--', c='k', lw=1.0)
    plt.title(latt.name.upper())
    plt.xlabel('T/T_m')
    plt.ylabel('Accuracy')
    plt.savefig(fnames.fig_tmplt.format(latt.name), dpi=300)
    plt.clf()

