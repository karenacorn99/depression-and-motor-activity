from feature_gen import *
from utils import *
from modeling import *
import pickle

from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    #create_and_save_subjects()

    # load pre-saved subjects
    with open('../data/subject.pkl', 'rb') as input:
        subjects =  pickle.load(input)

    X_raw, y = generate_data(subjects)
    X = get_features(X_raw)

    # KNN_pred = KNN(X, y)
    #LinearSVM_pred = Linear_SVC(X, y)
    #RBFSVM_pred = RBFSVM(X, y)
    DT_pred =  DecisionTree(X, y)
    acc = accuracy_score(y, DT_pred)
    print(acc)












