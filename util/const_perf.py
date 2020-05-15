import numpy as np

from . import dir_util, constants as cnst

perf_features = {}
for latt in cnst.lattices:
  x = np.loadtxt(dir_util.perf_features_path(latt, scaled=True))
  perf_features[latt.name] = x

