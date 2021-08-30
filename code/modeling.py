from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import *
from metrics import *
from collections import Counter
import numpy as np
from NN import training_loop

def run_model(X, y, model):
    model_func_dic = {'KNN': KNN, 'LinearSVC': LinearSVC, 'RBFSVM': RBFSVM, 'GaussianProcess': GaussianProcess,
                       'DecisionTree': DecisionTree, 'RandomForest': RandomForest, 'NeuralNet': NeuralNet,
                       'AdaBoost': AdaBoost, 'NaiveBayes': NaiveBayes, 'QDA': QDA, 'ZeroBaseline': ZeroRBaseline}

    total_performance = {'precision': np.array([0, 0, 0], dtype=np.float64),
     'recall/sensitivity': np.array([0, 0, 0], dtype=np.float64),
     'accuracy': np.array([0, 0, 0], dtype=np.float64),
     'specificity': np.array([0, 0, 0], dtype=np.float64),
     'Matthews correlation': np.array([0, 0, 0], dtype=np.float64),
     'f1': np.array([0, 0, 0], dtype=np.float64)}

    # 10-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model_func = model_func_dic[model]
        prediction = model_func(X_train, X_test, y_train)
        performance = get_performance(y_test, prediction)
        total_performance = {k: total_performance[k] + performance[k] for k, v in total_performance.items()}

    avg_performance = {k: v/10 for k, v in total_performance.items()}
    return avg_performance

def KNN(X_train, X_test, y_train):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return predictions

def LinearSVC(X_train, X_test, y_train):
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return predictions

def RBFSVM(X_train, X_test, y_train):
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return predictions

def GaussianProcess(X_train, X_test, y_train):
    kernel = 1.0 * RBF(1.0) # radial basis kernel
    clf = GaussianProcessClassifier(kernel=kernel, random_state=42).fix(X_train, y_train)
    predictions = clf.predict(X_test)
    return  predictions

def DecisionTree(X_train, X_test, y_train):
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return predictions

def RandomForest(X_train, X_test, y_train):
    clf = RandomForestClassifier(max_depth=2, random_state=42)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    return prediction

def NeuralNet(X_train, X_test, y_train):
    predictions = training_loop(X_train, y_train, X_test)
    return predictions

def AdaBoost(X_train, X_test, y_train):
    clf = AdaBoostClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return predictions

def NaiveBayes(X_train, X_test, y_train):
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return predictions

# Quadratic Discriminant Analysis
def QDA(X_train, X_test, y_train):
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return predictions

def ZeroRBaseline(X_train, X_test, y_train):
    vote_count = Counter(y_train)
    vote = max(vote_count, key=vote_count.get)
    return np.array([vote] * len(X_test))

