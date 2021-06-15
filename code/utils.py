import pickle
import pandas  as pd
from subject import subject
import numpy as np

def create_and_save_subjects():
    ##### initialize subjects from score.csv #####
    scores_df = pd.read_csv('../data/scores.csv')

    condition_df = scores_df[23:]
    mean_days = condition_df['days'].mean()
    print(mean_days)

    # replace nan & empty str with -1
    scores_df = scores_df.replace(np.nan, -1)
    scores_df = scores_df.replace(' ', -1)
    print(sum(scores_df['days']))
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

    ### correct & verify number of days in subject.days, values from scores.csv are incorrect ###
    # number of groups
    for s in subjects:
        num_of_group = len(s.motor_data_days)
        num_of_distinct_days = len(set(s.motor_data_df['date']))
        assert num_of_group == num_of_distinct_days
        s.days = num_of_distinct_days
    ### end of correction & verification ###

    # save subjects to file
    save_object(subjects, '../data/subject.pkl')

    return

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)