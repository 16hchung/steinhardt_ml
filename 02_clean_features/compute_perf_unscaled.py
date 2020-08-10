import numpy as np
from pathlib import Path
from collections import Counter

from util import constants as C
from util import dir_util as dirC

if __name__=='__main__':
  tmp_all_dir = Path(C.clean_feat_path) / 'tmp_perf' # TODO: you can delete this directory after running this
  tmp_all_dir.mkdir(exist_ok=True)
  # TODO vvv change lattices to be just your training lattices
  for latt in C.lattices:
    dest = dirC.perf_features_path(latt, scaled=False)
    in_template = dirC.all_features_path01(latt, pseudo=True, perfect=True)

    Xs = []
    # TODO vvv did we change the name of possible_n_neigh?
    for n_neigh in C.possible_n_neigh:
      in_file = in_template.format(n_neigh)
      Xs.append(np.loadtxt(in_file))
    X = np.concatenate(Xs, axis=1)

    tmp_file = str(tmp_all_dir / f'perf_{latt.name}_all.csv')
    np.savetxt(tmp_file, X, fmt='%.10e')

    # find most frequent line in saved perfect file
    with open(tmp_file) as f:
      cnts = Counter(l.strip() for l in f)
    most_common, _ = cnts.most_common(1)[0]
    with open(dest, 'w') as f:
      f.write(most_common)
