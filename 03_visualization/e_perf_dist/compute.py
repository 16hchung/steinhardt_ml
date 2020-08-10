import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from tqdm import tqdm

from util import dir_util
from util.const_perf import perf_features
from util import constants as cnst

def main(euc=True, cos=False): #, synth=False):
  if not euc and not cos:
    print('either euc or cos must be true')
    return

  #dir_suffix = '_synth' if synth else '_real'
  dir_suffix = '_synth_real'
  if euc: dir_suffix += '_euc'
  if cos: dir_suffix += '_cos'
  cut = 10000

  paths     = dir_util.clean_features_paths02(istest=True, pseudo=False)
  synth_paths = dir_util.clean_features_paths02(istest=False, pseudo=True)
  liq_paths = dir_util.clean_features_paths02(istest=True, pseudo=False, liq=True)
  X     = np.loadtxt(paths.X.format('concat_'))
  y     = np.loadtxt(paths.y.format('concat_'))
  X_syn = np.loadtxt(synth_paths.X.format('concat_'))
  y_syn = np.loadtxt(synth_paths.y.format('concat_'))
  X_liq = np.loadtxt(liq_paths.X.format('concat_'))
  np.random.shuffle(X_liq)
  X_liq = X_liq[:cut]
  
  for latt in tqdm(cnst.lattices):
    perfx = perf_features[latt.name]
    lattX = X[y==latt.y_label]
    synth_lattX = X_syn[y_syn==latt.y_label]
    np.random.shuffle(lattX)
    np.random.shuffle(synth_lattX)
    lattX = lattX[:cut]
    synth_lattX = synth_lattX[:cut]

    latt_dist = np.ones(len(lattX))
    synth_latt_dist = np.ones(len(synth_lattX))
    liq_dist  = np.ones(len(X_liq))
    if euc:
      latt_dist = latt_dist * np.linalg.norm(lattX - perfx, axis=-1)
      synth_latt_dist = synth_latt_dist * np.linalg.norm(synth_lattX - perfx, axis=-1)
      liq_dist  = liq_dist  * np.linalg.norm(X_liq - perfx, axis=-1)
    if cos:
      latt_dist = latt_dist * cosine_distances(lattX, np.expand_dims(perfx, axis=0))[:,0]
      synth_latt_dist = synth_latt_dist * cosine_distances(synth_lattX, np.expand_dims(perfx, axis=0))[:,0]
      liq_dist  = liq_dist  * cosine_distances(X_liq, np.expand_dims(perfx, axis=0))[:,0]

    #plt.hist([latt_dist, liq_dist],
    #         bins=100,
    #         density=True,
    #         label=[latt.name,'all liquid'],
    #         histtype='barstacked',
    #         alpha=.75)#,
             #label=[f'synth_{latt.name}', latt.name,'all liquid'])#,
             #histtype=['step', 'stepfilled', 'stepfilled'])
    plt.rcParams.update({'font.size': 16, 'figure.autolayout': True})
    bins = 50
    plt.hist(latt_dist, bins=bins, density=True, histtype='barstacked', label=latt.name.upper(), alpha=.5)
    plt.hist(liq_dist, bins=bins, density=True, histtype='barstacked', label='all liquid', alpha=.5)
    plt.hist(synth_latt_dist, bins=bins, color='r', density=True, histtype='step', label=f'synthetic {latt.name.upper()}', linewidth=2)

    # plot potential cutoffs
    #p99 = np.percentile(synth_latt_dist, 99)
    #p95 = np.percentile(synth_latt_dist, 95)
    p90 = np.percentile(synth_latt_dist, 90)
    #std, mean = np.std(synth_latt_dist), np.mean(synth_latt_dist)
    #std2     = mean + 2*std
    #std2half = mean + 2.5*std
    #std3     = mean + 3*std
    #lines = [p90, p95, p99, std2, std2half, std3]
    #colors = ['r', 'r', 'r', 'b', 'b', 'b']
    #styles = ['solid', 'dashed', 'dotted', 'solid', 'dashed', 'dotted']
    #lbls  = ['90th %ile', '95th %ile', '99th %ile', r'2$\sigma$', r'2.5$\sigma$', r'3$\sigma$']
    #for x, lbl, color, style in zip(lines, lbls, colors, styles):
    #  plt.axvline(x, color=color, linestyle=style, label=lbl)
    plt.axvline(p90, color='r', linestyle='dotted', label='90th %ile', linewidth=2.5)

    plt.legend(loc='upper right', fontsize='medium')
    plt.xlabel(rf'Distance from perfect features $x^*_{{{latt.name.upper()}}}$')
    plt.ylabel('Frequency')
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
  main(euc=args.euc, cos=args.cos)#, synth=args.synth)
