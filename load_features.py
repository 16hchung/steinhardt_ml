from constants import steinhardt_fname_tmpl, steinhardt_dir, structures, test_ts
import numpy as np

def get_train_steins():
    X_train = [np.loadtxt(steinhardt_fname_tmpl.format(steinhardt_dir, s, 'train')) for   [s,_,_] in structures]
    y_train = [l*np.ones(X_train[i].shape[0],dtype=int)                             for i,[_,_,l] in enumerate(structures)]
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    return X_train, y_train


