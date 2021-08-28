from feature_gen import *
from utils import *
from modeling import *
import pickle
from metrics import *
from sklearn.metrics import *
import argparse

from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    # create_and_save_subjects()

    # parse command line arguments
    parser = argparse.ArgumentParser(description="Process experiment configuration")
    parser.add_argument('--model', help='choose model')
    parser.add_argument('--features', help='select features to be used')
    args = vars(parser.parse_args())
    print("========== Start Experiment ==========")
    print('\n'.join("{:10} {}".format(k+':', v) for k, v in args.items()))

    # get list of features
    features = args['features'].split('+')
    # generate training data & label
    X, y = get_training_data(features)
    # train and evaluation
    result = run_model(X, y, args['model'])
    print("========== End of Experiment ==========")













