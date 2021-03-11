import pprint
import pandas as pd
from collections import Counter

class subject:
    def __init__(self, number, days, gender, age, afftype, melanch,
                 inpatient, edu, marriage, work, madrs1, madrs2):
        self.number = number
        self.days = days
        self.gender = gender
        self.age = age
        self.afftype = afftype
        self.melanch = melanch
        self.inpatient = inpatient
        self.edu = edu
        self.marriage = marriage
        self.work = work
        self.madrs1 = madrs1
        self.madrs2 = madrs2
        self.label = 'condition' if number[3] == 'd' else 'control'
        self.motor_data_df = None # motor data full df
        # {date: df with columns timestamp, activity}
        self.motor_data_days = None # motor data df by days

    def add_motor_data(self, file):
        # add full df
        motor_data_df = pd.read_csv(file)
        self.motor_data_df = motor_data_df
        self.motor_data_df['timestamp'] = self.motor_data_df['timestamp'].apply(lambda x : x.split()[1])
        # split full df by day
        self.motor_data_days = {k : self.motor_data_df.iloc[v, [0, 2]] for k, v in self.motor_data_df.groupby(['date']).groups.items()}
        return

    # plot a single day's motor data
    def view_motor_data_by_day(self, day=0):
        return

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        return 'subject({})'.format(', '.join(list(map(lambda x : str(x), self.__dict__.values()))))


