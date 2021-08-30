from sklearn.metrics import *
import numpy as np
from collections import Counter

def get_performance(y_true, y_pred):
    return {'precision': get_prec(y_true, y_pred),
            'recall/sensitivity': get_rec(y_true, y_pred),
            'accuracy': get_accuracy(y_true, y_pred),
            'specificity': get_spec(y_true, y_pred),
            'Matthews correlation': get_mcc(y_true, y_pred),
            'f1': get_f1(y_true, y_pred)}

# Precision
def get_prec(y_true, y_pred):
    class_0, class_1 = precision_score(y_true, y_pred, average=None)
    avg = precision_score(y_true, y_pred, average='weighted')
    return np.array([class_1, class_0, avg])

# Recall/Sensitivity
def get_rec(y_true, y_pred):
    class_0, class_1 = recall_score(y_true, y_pred, average=None)
    avg = recall_score(y_true, y_pred, average='weighted')
    return np.array([class_1, class_0, avg])

# Accuracy
def get_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return np.array([accuracy, accuracy, accuracy])

# Specificity: how many negative selected elements are truly negative?
def get_spec(y_true, y_pred):
    y_true = list(map(lambda x : 1-x, y_true))
    y_pred = list(map(lambda x : 1-x, y_pred))
    class_0, class_1 = recall_score(y_true, y_pred, average=None)
    avg = recall_score(y_true, y_pred, average='weighted')
    return np.array([class_1, class_0, avg])

# Matthews correlation coefficient: balanced account of TP, FP, TN and FN
def get_mcc(y_true, y_pred):
    mcc = matthews_corrcoef(y_true, y_pred)
    return np.array([mcc, mcc, mcc])

# F1-score: harmonic mean of precision and recall
def get_f1(y_true, y_pred):
    class_0 = f1_score(y_true, y_pred, pos_label=0)
    class_1 = f1_score(y_true, y_pred, pos_label=1)
    avg = f1_score(y_true, y_pred, average='weighted')
    return np.array([class_1, class_0, avg])

# F_beta: a more general F score, recall is considered beta times as important as precision