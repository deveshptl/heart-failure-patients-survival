from random import randint, uniform

values = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure',
          'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'DEATH_EVENT']
value_ranges = {
    'age': [40, 95],
    'anaemia': [0, 1],
    'creatinine_phosphokinase': [23, 7861],
    'diabetes': [0, 1],
    'ejection_fraction': [14, 80],
    'high_blood_pressure': [0, 1],
    'platelets': [25100, 850000],
    'serum_creatinine': [0.5, 9.4],
    'serum_sodium': [113, 148],
    'sex': [0, 1],
    'smoking': [0, 1],
    'time': [4, 285],
    'DEATH_EVENT': [0, 1]
}


def generateData(size, filename):
    f = open(f'data/{filename}', 'w')
    f.write(','.join(values))
    f.write('\n')

    for _ in range(size):
        row = generateRow()
        f.write(','.join(row))
        f.write('\n')


def generateRow():
    row = []
    for key in values:
        value = value_ranges[key]
        if key == 'serum_creatinine':
            row.append(str(uniform(0.5, 9.4)))
            continue
        row.append(str(randint(value[0], value[1])))
    return row


generateData(10000, 'auto_generated_data.csv')
