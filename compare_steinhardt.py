'''
load train steinhardts
load test steinhardts
loop through structures and print # predicted bcc in bcc for example

load test cartesians
loop through structures and print # predicted by ovito for ptm and cna
'''

from ovito.io import import_file
from ovito.modifiers import CommonNeighborAnalysisModifier as CNAModifier, \
                            PolyhedralTemplateMatchingModifier as PTMModifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from softmax import Softmax
import numpy as np

from constants import steinhardt_fname_tmpl, steinhardt_dir, structures, test_ts

X_train = [np.loadtxt(steinhardt_fname_tmpl.format(steinhardt_dir, s, 'train')) for   [s,_,_] in structures]
y_train = [l*np.ones(X_train[i].shape[0],dtype=int)                             for i,[_,_,l] in enumerate(structures)]
X_train = np.vstack(X_train)
y_train = np.hstack(y_train)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_train,y_train = shuffle(X_train,y_train)

# train svm based on steinhardt
#clf = SVC(C=1.0, max_iter=10000)
#clf.fit(X_train, y_train)
clf = Softmax()
clf.train(X_train, y_train)

def accuracy(pred, label):
    n_correct = np.count_nonzero(pred == label)
    return n_correct / pred.shape[0]

for [structure, fname, ov_label] in structures:
    print('\n###########################\nComparing {} identification...'.format(structure))
    test_cart_fname = fname.format(test_ts)
    test_stei_fname = steinhardt_fname_tmpl.format(steinhardt_dir, structure, 'val')

    # determine structure using common neighbor analysis
    cna_pipe = import_file(test_cart_fname)
    cna_pipe.modifiers.append(CNAModifier())
    output = cna_pipe.compute()
    cna_pred = output.particle_properties.structure_type.array
    print('CNA accuracy: {}'.format(accuracy(cna_pred, ov_label)))

    # determine structure using polyhedral template matching
    ptm_pipe = import_file(test_cart_fname)
    ptm_pipe.modifiers.append(PTMModifier())
    output = ptm_pipe.compute()
    ptm_pred = output.particle_properties.structure_type.array
    print('PTM accuracy: {}'.format(accuracy(ptm_pred, ov_label)))

    # load steinhardt for this structure and predict with svm
    X_val = np.loadtxt(test_stei_fname)
    X_val = scaler.transform(X_val)
    svm_pred = clf.predict(X_val)
    print('Steinhardt accuracy: {}'.format(accuracy(svm_pred, ov_label)))
