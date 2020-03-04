import numpy as np
import matplotlib.pyplot as plt

from util import dir_util, constants as cnst

def run(d_fxn_path):
  for latt in cnst.lattices:
    df = np.loadtxt(d_fxn_path.data_tmplt.format(latt.name))
    plt.hist(df, alpha=.5, edgecolor='k')
    plt.savefig(d_fxn_path.fig_tmplt.format(latt.name))
    plt.clf()
