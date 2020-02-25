import numpy as np
import matplotlib.pyplot as plt

from util import dir_util
from util import constants as cnst

def plot_mins(synth_zscores, zscores, y, fnames, latt):
  mins       = zscores.min(axis=1)
  synth_mins = synth_zscores.min(axis=1)
  plot_hist(synth_mins, mins, y, fnames.mins, latt)

def plot_maxs(synth_zscores, zscores, y, fnames, latt):
  maxs       = zscores.max(axis=1)
  synth_maxs = synth_zscores.max(axis=1)
  plot_hist(synth_maxs, maxs, y, fnames.maxs, latt)

def plot_avgs(synth_zscores, zscores, y, fnames, latt):
  avgs       = np.mean(zscores, axis=1)
  synth_avgs = np.mean(synth_zscores, axis=1)
  plot_hist(synth_avgs, avgs, y, fnames.avgs, latt)

def plot_meds(synth_zscores, zscores, y, fnames, latt):
  meds       = np.median(zscores, axis=1)
  synth_meds = np.median(synth_zscores, axis=1)
  plot_hist(synth_meds, meds, y, fnames.meds, latt)

def plot_hist(synth_values, values, y, fname, wrt_latt):
  minZ = min(synth_values.min(), values.min())
  maxZ = max(synth_values.min(), values.max())
  bins = np.linspace(minZ, maxZ, 20)
  plt.hist(synth_values, bins=bins, alpha=.5, label=wrt_latt.name+'*', edgecolor='k')
  for latt in cnst.lattices:
    plt.hist(values[y==latt.y_label], bins=bins, alpha=.5, label=latt.name, edgecolor='k')
  plt.legend(loc='upper right')
  plt.savefig(fname)
  plt.clf()

def main():
  for latt in cnst.lattices:
    synth_zscores = np.loadtxt(dir_util.zscore_data_path03(latt, synth=True))
    zscores = np.loadtxt(dir_util.zscore_data_path03(latt))
    y = np.loadtxt(dir_util.clean_features_paths02(istest=True).y.format(latt.n_neigh))
    save_paths = dir_util.zscore_fig_path03(latt)

    plot_mins(synth_zscores, zscores, y, save_paths, latt)
    plot_avgs(synth_zscores, zscores, y, save_paths, latt)
    plot_maxs(synth_zscores, zscores, y, save_paths, latt)
    plot_meds(synth_zscores, zscores, y, save_paths, latt)

if __name__=='__main__':
  main()
