import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib

################################################################################
# Train model.                                                                 #
################################################################################
def run(X,y, model, model_params, model_path, scores_path):
  params = {'tol':1e-3,'max_iter':1000}
  params.update(model_params)
  
  # Split data (train on 50,000 points)
  X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.2, shuffle=False)

  # Fit on train set.
  clf = model(**params)
  clf.fit(X_train,y_train)
  joblib.dump(clf,model_path)

  ################################################################################
  # Compute validation set accuracy and confusion matrix.                        #
  ################################################################################

  with open(scores_path, 'w') as f:
    acc_valid = clf.score(X_valid,y_valid) # Accuracy on the validation set.
    f.write('Accuracy on validation set: %.2f\n\n' % (acc_valid*100))
    f.write('%s\n' % classification_report(y_valid,clf.predict(X_valid)))
    f.write('Confusion Matrix:\n')
    f.write('%s' % confusion_matrix(y_valid,clf.predict(X_valid)))

################################################################################
