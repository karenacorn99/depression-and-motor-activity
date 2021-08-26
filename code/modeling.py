from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree

def run_model(X, y, model):
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

