import numpy as np

def generate_data(subjects):
    X_raw = []
    y = []
    for s in subjects:
        for day, df in s.motor_data_days.items():
            X_raw.append(df['activity'].tolist())
            y.append(s.label)
    assert len(X_raw) == len(y)
    return (X_raw, y)

def get_features(X_raw):
    mean = get_mean(X_raw)
    std = get_std(X_raw)
    num_zero_activity = get_num_zero_activity(X_raw)
    # concatenate elementwise
    feature_vectors = np.array([m + s + n for m, s, n in zip(mean, std, num_zero_activity)])
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

# percenrage of events with no activity ie activity level = 0
def get_num_zero_activity(X_raw):
    return list(map(lambda x : [sum(np.array(x) == 0) / len(x)], X_raw))