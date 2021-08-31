from feature_gen import *
from modeling import *
import argparse


if __name__ == '__main__':

    # create_and_save_subjects()

    # python __main__.py --model KNN --features activity_mean

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
    print('\n'.join("{:30} {}".format(k+':', v) for k, v in result.items()))
    print("========== End of Experiment ==========")













