import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from tqdm import tqdm

from util import dir_util
from util.const_perf import perf_features
from util import constants as cnst

def main(euc=True, cos=False, synth=False):
  if not euc and not cos:
    print('either euc or cos must be true')
    return

  dir_suffix = '_synth' if synth else '_real'
  if euc: dir_suffix += '_euc'
  if cos: dir_suffix += '_cos'

  paths     = dir_util.clean_features_paths02(istest=not synth, pseudo=synth)
  liq_paths = dir_util.clean_features_paths02(istest=True, pseudo=False, liq=True)
  X     = np.loadtxt(paths.X.format('concat_'))
  y     = np.loadtxt(paths.y.format('concat_'))
  X_liq = np.loadtxt(liq_paths.X.format('concat_'))
  y_liq = np.loadtxt(liq_paths.y.format('concat_'))
  
  for latt in tqdm(cnst.lattices):
    perfx = perf_features[latt.name]
    lattX = X[y==latt.y_label][:]
    latt_dist = np.ones(len(lattX))
    liq_dist  = np.ones(len(X_liq))
    if euc:
      latt_dist = latt_dist * np.linalg.norm(lattX - perfx, axis=-1)
      liq_dist  = liq_dist  * np.linalg.norm(X_liq - perfx, axis=-1)
    if cos:
      latt_dist = latt_dist * cosine_distances(lattX, np.expand_dims(perfx, axis=0))
      liq_dist  = liq_dist  * cosine_distances(X_liq, np.expand_dims(perfx, axis=0))

    plt.hist([latt_dist, liq_dist], bins=100, density=True, label=[latt.name,'liquid'])

    # plot potential cutoffs
    p99 = np.percentile(latt_dist, 99)
    p95 = np.percentile(latt_dist, 95)
    p90 = np.percentile(latt_dist, 90)
    std, mean = np.std(latt_dist), np.mean(latt_dist)
    std2     = mean + 2*std
    std2half = mean + 2.5*std
    std3     = mean + 3*std
    lines = [p90, p95, p99, std2, std2half, std3]
    colors = ['r', 'r', 'r', 'b', 'b', 'b']
    styles = ['solid', 'dashed', 'dotted', 'solid', 'dashed', 'dotted']
    lbls  = ['90th %ile', '95th %ile', '99th %ile', r'2$\sigma$', r'2.5$\sigma$', r'3$\sigma$']
    for x, lbl, color, style in zip(lines, lbls, colors, styles):
      plt.vlines(x, 0, 1, colors=color, linestyles=style, label=lbl)
    #plt.vlines(lines, 0, 1, label=lbls)

    plt.legend()
    fig_path = dir_util.perf_dist_fig_path03(latt, dir_suffix)
    plt.savefig(fig_path, dpi=300)

    plt.clf()

if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--euc', action='store_true')
  parser.add_argument('--cos', action='store_true')
  parser.add_argument('--synth', action='store_true')
  args = parser.parse_args()
  main(euc=args.euc, cos=args.cos, synth=args.synth)
