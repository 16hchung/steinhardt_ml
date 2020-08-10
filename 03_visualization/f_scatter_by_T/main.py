from sklearn.utils import shuffle
from tqdm import tqdm
import pandas as pd
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt

from util import constants as C, dir_util

def plotT_gradients():
  cut = 500

  paths = dir_util.pca_data_paths03()
  with open(paths.pca, 'rb') as f:
    pca = pk.load(f)

  for latt in tqdm(C.lattices[:]):
    pcaXs = []
    temps = []
    for temp in tqdm(range(latt.low_temp, latt.high_temp+latt.step_temp, latt.step_temp)):
      paths = dir_util.clean_features_paths02(istest=True, lattice=latt, temp=temp)
      X = shuffle(np.loadtxt(paths.X.format('concat_')))[:cut, :]
      pcaX = pca.transform(X)
      pcaXs.append(pcaX)
      temps.append(np.full(len(pcaX),temp))
    pcaX = np.vstack(pcaXs)
    T = np.concatenate(temps)
    T = T / latt.T_m
    plt.rcParams.update({'font.size': 16, 'figure.autolayout': True})
    plt.scatter(pcaX[:,0], pcaX[:,1], c=T, alpha=.3, marker='.', linewidths=0)
    cbar = plt.colorbar()
    cbar.set_label(r'$T/T_m$')
    perf = np.loadtxt(dir_util.perf_features_path(latt, scaled=True))[:]
    pca_perf = pca.transform(np.array([perf]))
    plt.scatter(pca_perf[:,0], pca_perf[:,1], s=180, c='r', edgecolors='w', linewidths=1, marker='*', alpha=1, label='perfect features')
    #plt.scatter(pca_perf[:,0], pca_perf[:,1], markersize=10, c='r', edgecolors='k', linewidths=1, marker='*', alpha=1, label='perfect features')
    plt.legend()
    plt.title(f'PCA of {latt.name.upper()} Features by Temperature')
    plt.xlabel('First PCA Component')
    plt.ylabel('Second PCA Component')
    plt.savefig(f'{C.vis_figures_path}pcaByT_{latt.name}.png', dpi=300)
    plt.clf()

if __name__=='__main__':
  plotT_gradients()
