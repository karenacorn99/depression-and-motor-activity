import numpy as np
import pickle

def get_training_data(features):
    subjects = get_subjects()
    X_raw, y = generate_data(subjects)
    # convert y to numeric label
    y = list(map(lambda x : 1 if x == 'condition' else 0, y))
    X = get_features(X_raw, features)
    return (X, y)

# load pre-saved subjects from saved file
def get_subjects():
    with open('../data/subject.pkl', 'rb') as input:
        subjects = pickle.load(input)
    return subjects

def generate_data(subjects):
    X_raw = []
    y = []
    for s in subjects:
        for day, df in s.motor_data_days.items():
            X_raw.append(df['activity'].tolist())
            y.append(s.label)
    assert len(X_raw) == len(y)
    return (X_raw, y)

def get_features(X_raw, features):

    feature_func_dic = {'mean': get_mean, 'std': get_std, 'num_zero_activity': get_num_zero_activity}

    feature_vectors = []

    for f in features:
        feature_function = feature_func_dic[f]
        current_feature_vector = feature_function(X_raw)
        print(current_feature_vector)
        feature_vectors.append(current_feature_vector)
    print(feature_vectors)

    # # concatenate elementwise
    # feature_vectors = np.array([m + s + n for m, s, n in zip(mean, std, num_zero_activity)])
    # normalize
    feature_vectors = feature_vectors - feature_vectors.mean(axis=0, keepdims=True)
    feature_vectors = feature_vectors / (feature_vectors.std(axis=0) + 1e-10) # prevent division by 0
    return feature_vectors

# mean activity level
def get_mean(X_raw):
    return list(map(lambda x : [np.mean(x)], X_raw))

# standard deviation
def get_std(X_raw):
    return list(map(lambda x : [np.std(x)], X_raw))

# percentage of events with no activity ie activity level = 0
def get_num_zero_activity(X_raw):
    return list(map(lambda x : [sum(np.array(x) == 0) / len(x)], X_raw))