import pandas as pd
import pprint
from collections import Counter

def inspect_scores_csv():
    scores_file = '../data/scores.csv'
    scores_df = pd.read_csv(scores_file)
    desc_str = ''
    desc_str += '=== scores.csv ===\n'
    desc_str += '{} rows\n'.format(len(scores_df))
    # columns info
    desc_str += '{} columns: \n{}\n'.format(len(scores_df.columns), list(scores_df.columns))
    column_info = {'number': 'patient identifier',
                   'days': 'number of days of measurement',
                   'gender': '1: female, 2: male',
                   'age': 'age group of patient',
                   'afftype': '1: bipolar II, 2: unipolar depressive, 3: bipolar I',
                   # I: full manic episode II: major depressive episode
                   # https://www.healthline.com/health/bipolar-disorder/bipolar-1-vs-bipolar-2#symptoms
                   'melanch': 'melancholia, 1 for yes, 3 for no', # extreme despair and guilt, subtype of depression
                   'inpatient': '1: inpatient, 2: outpatient',
                   'edu': 'education grouped in years',
                   'marriage': '1: married or cohabiting, 2: single',
                   'work': '1: working or studying, 2: unemployed/sick leave/pension',
                   'madrs1': 'MADRS score when measurement started',
                   'madrs2': 'MADRS when measurement stopped'
                   # 0-6: no depression 7-19: mild 20-34: moderate >35: severe
                   }
    column_info_str = pprint.pformat(column_info)
    desc_str += (column_info_str + '\n')
    desc_str += 'days\n' + pprint.pformat(sorted(Counter(scores_df['days']).items())) + '\n'
    desc_str += 'gender\n' + pprint.pformat(sorted(Counter(scores_df['gender']).items())) + '\n'
    desc_str += 'age\n' + pprint.pformat(sorted(Counter(scores_df['age']).items())) + '\n'
    condition_df = scores_df.iloc[:23, :]
    assert len(condition_df) == 23
    control_df = scores_df.iloc[23:, :]
    assert len(control_df) == 32
    desc_str += 'total # of days condition: {}\n'.format(sum(condition_df['days']))
    desc_str += 'total # of days control: {}\n'.format(sum(control_df['days']))
    desc_str += 'afftype\n' + pprint.pformat(sorted(Counter(condition_df['afftype']).items())) + '\n'
    desc_str += 'melanch\n' + pprint.pformat(sorted(Counter(condition_df['melanch']).items())) + '\n'
    desc_str += 'inpatient\n' + pprint.pformat(sorted(Counter(condition_df['inpatient']).items())) + '\n'
    desc_str += 'edu\n' + pprint.pformat(sorted(Counter(condition_df['edu']).items())) + '\n'
    desc_str += 'marriage\n' + pprint.pformat(sorted(Counter(condition_df['marriage']).items())) + '\n'
    desc_str += 'work\n' + pprint.pformat(sorted(Counter(condition_df['work']).items())) + '\n'
    desc_str += 'madrs1\n' + pprint.pformat(sorted(Counter(condition_df['madrs1']).items())) + '\n'
    desc_str += 'madrs2\n' + pprint.pformat(sorted(Counter(condition_df['madrs2']).items())) + '\n'
    # control individuals only have number, days, gender and age
    # condition_22 has empty string for edu
    return desc_str

if __name__ == '__main__':
    score_desc = inspect_scores_csv()
    with open('../record/desc_score_csv.txt', 'w') as f:
        f.write(score_desc)