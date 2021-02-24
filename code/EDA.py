import pandas as pd
import pprint
from collections import Counter

def inspect_scores_csv():
    scores_file = '../data/scores.csv'
    scores_df = pd.read_csv(scores_file)
    print('=== scores.csv ===')
    print('{} rows'.format(len(scores_df)))
    print('{} columns: \n{}'.format(len(scores_df.columns), list(scores_df.columns)))
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
    #pprint.pprint(column_info)
    pprint.pprint(sorted(Counter(scores_df['days']).items()))

    return


if __name__ == '__main__':
    inspect_scores_csv()