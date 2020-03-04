import numpy as np
from pathlib import Path
import joblib

from util import dir_util, constants as cnst

def run(X, y, model, model_params, d_fxn_path, ms_paths, pretrained):
  params = {'tol':1e-3,'max_iter':1000}
  params.update(model_params)

  model_path = ms_paths.model_tmplt.format(hyperprm_sffx='')
  if Path(model_path).exists() and pretrained:
    clf = joblib.load(model_path)
  else:
    clf = model(**params)
    clf.fit(X,y)

  for latt in cnst.lattices:
    lattX = X[y==latt.y_label]
    df = clf.decision_function(lattX)[:, latt.y_label-1]
    np.savetxt(d_fxn_path.data_tmplt.format(latt.name), df)


