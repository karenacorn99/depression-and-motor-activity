import pandas as pd
import numpy as np
from subject import subject
from feature_gen import *

if __name__ == '__main__':
    ### initialize subjects from score.csv ###
    scores_df = pd.read_csv('../data/scores.csv')
    # replace nan & empty str with -1
    scores_df = scores_df.replace(np.nan, -1)
    scores_df = scores_df.replace(' ', -1)
    # check there is no nan
    assert scores_df.isnull().sum().sum() == 0

    subjects = [subject(row.number, row.days, row.gender, row.age, row.afftype, row.melanch,
                        row.inpatient, row.edu, row.marriage, row.work, row.madrs1, row.madrs2)
                for row in scores_df.itertuples()]
    #for s in subjects:
        #print(s)

    # add motor data
    for s in subjects:
        file = '../data/' + s.label + '/' + s.number + '.csv'
        s.add_motor_data(file)

    X_raw, y = generate_data(subjects)
    print(get_features(X_raw))








