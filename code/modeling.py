from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import *

def run_model(X, y, model):
    total_performance = {'precision': 0,
     'recall/sensitivity': 0,
     'accuracy': 0,
     'specificity': 0,
     'Matthews correlation': 0,
     'f1': 0}

    # 10-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    avg_performance = {k: v/10 for k, v in total_performance.items()}
    return

def KNN(X, y):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X, y)
    predictions = clf.predict(X)
    return predictions

def LinearSVC(X, y):
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    predictions = clf.predict(X)
    return predictions

def RBFSVM(X, y):
    clf = svm.SVC(kernel='rbf')
    clf.fit(X, y)
    predictions = clf.predict(X)
    return predictions

def GaussianProcess(X, y):
    return

def DecisionTree(X, y):
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)
    predictions = clf.predict(X)
    return predictions

def RandomForest(X, y):

    return

def NeuralNet(X, y):
    return

def AdaBoost(X, y):
    return

def NaiveBayes(X, y):
    return

def QDA(X, y):
    return

def ZeroRBaseline(X, y):
    return

