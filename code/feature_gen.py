import numpy as np
import pickle
import pandas as pd

full_timestamps = ["{:02d}:{:02d}:00".format(hour, minute) for hour in range(24) for minute in range(60)]

def get_training_data(features):
    subjects = get_subjects()
    X_raw, y = generate_data(subjects)
    # convert y to numeric label
    y = np.array(list(map(lambda x : 1 if x == 'condition' else 0, y)))
    X = get_features(X_raw, features)
    return (X, y)

# load pre-saved subjects from saved file
def get_subjects():
    with open('../data/subject.pkl', 'rb') as input:
        subjects = pickle.load(input)
    return subjects

def generate_data(subjects):
    assert len(full_timestamps) == 24 * 60

    # fill missing timestamps with None
    X_raw = []
    y = []
    for s in subjects:
        for day, df in s.motor_data_days.items():
            df.set_index("timestamp", inplace=True)
            df = df.reindex(full_timestamps).fillna(-1)
            X_raw.append(df['activity'].tolist())
            assert len(df['activity']) == 24*60
            y.append(s.label)
    assert len(X_raw) == len(y)
    return (X_raw, y)

def get_features(X_raw, features):

    # feature_len_dic = {'mean': 1, 'std': 1, 'num_zero_activity': 1}
    feature_func_dic = {'mean': get_mean, 'std': get_std, 'num_zero_activity': get_num_zero_activity,
                        'hour_mean': get_hour_mean, 'activity_mean': get_activity_mean, 'sleep_pattern': get_sleep_pattern}

    # feature_vector_length = np.sum([feature_len_dic.get(f) for f in features])
    feature_vectors = []

    for index, f in enumerate(features):
        feature_function = feature_func_dic[f]
        current_feature_vector = feature_function(X_raw)
        feature_vectors.append(np.array(current_feature_vector))
    #  concatenate element wise
    feature_matrix = np.concatenate(feature_vectors, axis=1)

    # normalize each feature
    feature_matrix = np.apply_along_axis(lambda x: x - np.mean(x), 0, feature_matrix)
    feature_matrix = np.apply_along_axis(lambda x: (x / (np.std(x))) + 1e-10, 0, feature_matrix) # prevent division by 0

    return feature_matrix

# mean activity level
def get_mean(X_raw):
    # use masked array to ignore -1
    means = np.apply_along_axis(lambda x: [np.mean(np.ma.array(x, mask=(x==-1)))], 1, np.array(X_raw))
    return np.nan_to_num(means, nan=0.0)

# standard deviation
def get_std(X_raw):
    # use masked array to ignore -1
    return np.apply_along_axis(lambda x: [np.std(np.ma.array(x, mask=(x==-1)))], 1, np.array(X_raw))

# percentage of events with no activity ie activity level = 0
def get_num_zero_activity(X_raw):
    # use masked array to ignore -1
    return np.apply_along_axis(lambda x: [np.sum(np.ma.array(x, mask=(x==-1))) / np.sum(x!=-1)], 1, np.array(X_raw))

def get_hour_mean(X_raw):
    # fill all missing values with 0
    X = np.array(X_raw)
    X[X==-1] = 0
    print(np.array(X_raw).shape)
    # group into hours
    X = np.array(X_raw).reshape(-1, 24, 60)
    X = np.apply_along_axis(np.mean, 2, X)
    return X

def get_activity_mean(X_raw):
    # avg of non-zero activity level
    # how much do you move every time you do stand up?
    X = np.array(X_raw, dtype=np.float64)
    X[X==0] = -1
    return get_mean(X)

def get_sleep_pattern(X_raw, threshold=300):
    return np.apply_along_axis(get_sleep_pattern_from_array, 1, np.array(X_raw))

def get_sleep_pattern_from_array(x, threshold=70):
    # reset timestamp, start from noon, end at 11:59am
    x = np.roll(x, 12 * 60)
    rolled_timestamps = np.roll(np.array(full_timestamps), 12 * 60)
    # convert to binary array
    x = np.where(x <= threshold, 1, 0)
    max_count, start_index, end_index = get_longest_consecutive_ones(x)
    #TODO: convert to numerical values
    return [max_count, rolled_timestamps[start_index], rolled_timestamps[end_index]]

def get_longest_consecutive_ones(x):
    max_count, cur_count = 0, 0
    max_end_index = 0

    for index, num in enumerate(x):
        if num == 1:
            cur_count += 1
        elif num == 0:
            if cur_count > max_count:
                max_count = cur_count
                max_end_index = index - 1
            cur_count = 0

    return (max_count, max_end_index - max_count + 1, max_end_index)

