from sklearn.metrics import *
import numpy as np

# Precision
def get_prec(y_true, y_pred):
    class_0, class_1 = precision_score(y_true, y_pred, average=None)
    avg = precision_score(y_true, y_pred, average='weighted')
    return (class_0, class_1, avg)

# Recall/Sensitivity
def get_rec(y_true, y_pred):
    class_0, class_1 = recall_score(y_true, y_pred, average=None)
    avg = recall_score(y_true, y_pred, average='weighted')
    return (class_0, class_1, avg)

# Accuracy
def get_accuracy(y_true, y_pred):
    accuracy = get_accuracy(y_true, y_pred)
    return (accuracy, accuracy, accuracy)

# Specificity: how many negative selected elements are truly negative?
def get_spec(y_true, y_pred):
    y_true = list(map(lambda x : 1-x, y_true))
    y_pred = list(map(lambda x : 1-x, y_pred))
    class_0, class_1 = recall_score(y_true, y_pred, average=None)
    avg = recall_score(y_true, y_pred, average='weighted')
    return (class_0, class_1, avg)

# Matthews correlation coefficient: balanced account of TP, FP, TN and FN
def get_mcc(y_true, y_pred):
    mcc = matthews_corrcoef(y_true, y_pred)
    return (mcc, mcc, mcc)

# F1-score: harmonic mean of precision and recall
def get_f1(y_true, y_pred):
    class_0 = f1_score(y_true, y_pred, pos_label=0)
    class_1 = f1_score(y_true, y_pred, pos_label=1)
    avg = f1_score(y_true, y_pred, average='weighted')
    return (class_0, class_1, avg)

# F_beta: a more general F score, recall is considered beta times as important as precision