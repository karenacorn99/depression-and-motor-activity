import numpy as np
import pickle
import pandas as pd

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
    full_timestamps = ["{:02d}:{:02d}:00".format(hour, minute) for hour in range(24) for minute in range(60)]
    assert len(full_timestamps) == 24 * 60

    # fill missing timestamps with None
    X_raw = []
    y = []
    for s in subjects:
        for day, df in s.motor_data_days.items():
            df.set_index("timestamp", inplace=True)
            df = df.reindex(full_timestamps).fillna(-1)
            X_raw.append(df['activity'].tolist())
            y.append(s.label)
    assert len(X_raw) == len(y)
    return (X_raw, y)

def get_features(X_raw, features):

    # feature_len_dic = {'mean': 1, 'std': 1, 'num_zero_activity': 1}
    feature_func_dic = {'mean': get_mean, 'std': get_std, 'num_zero_activity': get_num_zero_activity}

    # feature_vector_length = np.sum([feature_len_dic.get(f) for f in features])
    feature_vectors = []

    for index, f in enumerate(features):
        feature_function = feature_func_dic[f]
        current_feature_vector = feature_function(X_raw)
        feature_vectors.append(np.array(current_feature_vector))
    #  concatenate element wise
    feature_matrix = np.concatenate(feature_vectors, axis=1)
    print(np.array(feature_matrix.shape))

    # normalize each feature
    feature_matrix = np.apply_along_axis(lambda x: x - np.mean(x), 0, feature_matrix)
    feature_matrix = np.apply_along_axis(lambda x: (x / (np.std(x))) + 1e-10, 0, feature_matrix) # prevent division by 0

    return feature_matrix

# mean activity level
def get_mean(X_raw):
    # use masked array to ignore -1
    return np.apply_along_axis(lambda x: [np.mean(np.ma.array(x, mask=(x==-1)))], 1, np.array(X_raw))

# standard deviation
def get_std(X_raw):
    # use masked array to ignore -1
    return np.apply_along_axis(lambda x: [np.std(np.ma.array(x, mask=(x==-1)))], 1, np.array(X_raw))

# percentage of events with no activity ie activity level = 0
def get_num_zero_activity(X_raw):
    # use masked array to ignore -1
    return np.apply_along_axis(lambda x: [np.sum(np.ma.array(x, mask=(x==-1))) / np.sum(x!=-1)], 1, np.array(X_raw))