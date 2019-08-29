from sklearn.svm import SVC
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

from load_features import get_train_steins

def plot_curve(curve_fxn, extra_curve_args, model, title, xlabel, ylabel, fname, X, y, train_score_idx, test_score_idx, xvals):
    plt.figure()
    plt.title(title)
    plt.ylim(0.0, 1.1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # include standard scalar in estimator
    pipe = Pipeline(steps=[
        ('scale', StandardScaler()),
        ('clf', model)
    ])
    #import pdb;pdb.set_trace()

    result = curve_fxn(pipe, X, y, cv=5, n_jobs=1, scoring='accuracy', **extra_curve_args)

    train_scores = result[train_score_idx]
    test_scores  = result[test_score_idx]
    if isinstance(xvals, int):
        xvals = result[xvals]

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    print(test_scores_mean)
    plt.fill_between(xvals, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="r")
    plt.fill_between(xvals, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="g")
    plt.plot(xvals, train_scores_mean, 'o-', color="r",
           label="Training score")
    plt.plot(xvals, test_scores_mean, 'o-', color="g",
           label="Cross-validation score")
    plt.legend(loc='best')
    plt.savefig(fname)
    plt.clf()

X,y = get_train_steins()

# plot validation curve
param_range = [10,20,30,40,50]
extra_args = {'param_range':param_range, 'param_name':'clf__C'}
plot_curve(
    validation_curve, extra_args, SVC(), 
    'Validation curve', 'C', 'accuracy', 'kernel_svm_val_curve_finer.png', 
    X, y, 
    0, 1, param_range
)

# plot learning curve
