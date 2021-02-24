import pprint

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
        self.type = 'condition' if number[3] == 'd' else 'control'

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        return 'subject({})'.format(', '.join(list(map(lambda x : str(x), self.__dict__.values()))))


