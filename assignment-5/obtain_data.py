# import pandas as pd

# df = pd.read_csv('crime.csv', engine='python')
# df = df.drop(columns=['INCIDENT_NUMBER', 'OFFENSE_CODE', 'OFFENSE_DESCRIPTION', 'Lat', 'Long', 'Location', 'STREET', 'REPORTING_AREA', 'SHOOTING', 'UCR_PART', 'OCCURRED_ON_DATE'])
# df.to_csv('fixed_crime.csv', index=False)

import tkinter
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt

# crime.csv에서 데이터를 읽어옵니다. 이 읽어온 데이터를 dataframe 이라고 합니다.
# dataframe에 df라고 이름을 붙여줍니다.
df = pd.read_csv('crime.csv', engine='python')

# dataframe에서 YEAR의 내용을 기준으로 그룹화하고 (groupby)
# 각 그룹의 갯수를 센 다음 (size)
# 막대그래프로 그려줍니다 (plot)
df.groupby('YEAR').size().plot(kind='barh')

# plot 메소드는 내부적으로 matplotlib를 통해 그래프를 그려줍니다.
# 그래서 plt.show()를 통해 그래프가 보이도록 합시다.
plt.show()
# import pandas as pd

# df = pd.read_csv('fixed_crime.csv', engine='python')
# df = df.dropna(axis=0)
# df.to_csv('fixed_crime2.csv', index=False)



# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv('fixed_crime2.csv', engine='python')
# df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False).plot(kind='barh')
# plt.show()



# import pandas as pd

# df = pd.read_csv('fixed_crime2.csv', engine='python')
# print(*df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False).index.tolist(), sep='\n')



# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv('fixed_crime2.csv', engine='python')
# value_counts = df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False).tolist()
# value_counts_index = df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False).index.tolist()

# other_list = []

# for i in range(len(value_counts)):
#     if value_counts[i] <= 1000:
#         other_list.append(value_counts_index[i])

# def func(x):
#     if x in other_list:
#         return 'Other'
#     else:
#         return x
        
# df['OFFENSE_CODE_GROUP'] = df.apply(lambda x: func(x['OFFENSE_CODE_GROUP']), axis=1)

# df.to_csv('fixed_crime3.csv', index=False)

# df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False).plot(kind='barh')
# plt.show()

# print(len(df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False).index.tolist()))



# import pandas as pd
# import matplotlib.pyplot as plt

# accident_list = ['Medical Assistance', 'Motor Vehicle Accident Response', 'Fire Related Reports', 'Police Service Incidents']
# theft_list = ['Larceny', 'Larceny From Motor Vehicle', 'Property Lost', 'Residential Burglary', 'Auto Theft', 'Robbery', 'Fraud', 'Confidence Games', 'Commercial Burglary', 'Auto Theft Recovery']
# misdemeanor_list = ['Drug Violation', 'Liquor Violation', 'Vandalism', 'Towed', 'Violations', 'Disorderly Conduct', 'Firearm Violations', 'License Violation', 'Restraining Order Violations', 'Counterfeiting']
# service_list = ['Investigate Person', 'Investigate Property', 'Warrant Arrests', 'Missing Person Located', 'Property Found', 'Missing Person Reported', 'Recovered Stolen Property']
# violence_list = ['Verbal Disputes', 'Simple Assault', 'Aggravated Assault', 'Harassment']
# other_list = ['Other']

# df = pd.read_csv('fixed_crime3.csv', engine='python')

# def func(x):
#     if x in accident_list:
#         return 'Accident'
#     if x in theft_list:
#         return 'Theft'
#     if x in misdemeanor_list:
#         return 'Misdemeanor'
#     if x in service_list:
#         return 'Service'
#     if x in violence_list:
#         return 'Violence'
#     if x in other_list:
#         return 'Other'

# df['CLASSIFICATION'] = df.apply(lambda x: func(x['OFFENSE_CODE_GROUP']), axis=1)

# df.to_csv('fixed_crime4.csv', index=False)

# df['CLASSIFICATION'].value_counts(sort=True, dropna=False).plot(kind='barh')
# plt.show()




# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# df = pd.read_csv('fixed_crime4.csv', engine='python')

# drop_list = ['Accident', 'Theft', 'Misdemeanor', 'Service', 'Violence']

# new_df = df[df['CLASSIFICATION'] == 'Other']
# other_count = len(df[df['CLASSIFICATION'] == 'Other'])

# for class_name in drop_list:
#     class_index = df[df['CLASSIFICATION'] == class_name].index
#     under_sample_index = np.random.choice(class_index, other_count, replace=False)
#     under_sample = df.loc[under_sample_index]
#     new_df = new_df.append(under_sample)

# new_df = new_df.sample(frac=1).reset_index(drop=True)

# new_df.to_csv('fixed_crime5.csv', index=False)

# new_df['CLASSIFICATION'].value_counts(sort=True, dropna=False).plot(kind='barh')
# plt.show()



import pandas as pd

df = pd.read_csv('fixed_crime5.csv', engine='python')

def func_time(x):
    if 6 <= x and x < 12:
        return 'Morning'
    if 12 <= x and x < 17:
        return 'Afternoon'
    if 5 <= x and x < 19:
        return 'Evening'
    return 'Night'

df['TIME'] = df.apply(lambda x: func_time(x['HOUR']), axis=1)

def func_weekday(x):
    if x == 'Sunday' or x == 'Saturday':
        return 1
    else:
        return 0

df['WEEKDAY'] = df.apply(lambda x: func_weekday(x['DAY_OF_WEEK']), axis=1)

human_loss = ['Motor Vehicle Accident Response', 'Medical Assistance', 'Drug Violation', 'Simple Assault', 'Verbal Disputes', 'Aggravated Assault', 'Residential Burglary', 'Robbery', 'Harassment']
financial_loss = ['Motor Vehicle Accident Response', 'Larceny', 'Vandalism', 'Larceny From Motor Vehicle', 'Property Lost', 'Fraud', 'Residential Burglary', 'Auto Theft', 'Robbery', 'Confidence Games', 'Fire Related Reports', 'Counterfeiting', 'Commercial Burglary', 'Auto Theft Recovery']
inside = ['Vandalism', 'Commercial Burglary']
exist_victim_offender = ['Larceny', 'Simple Assault', 'Verbal Disputes', 'Larceny From Motor Vehicle', 'Property Lost', 'Aggravated Assault', 'Fraud', 'Residential Burglary', 'Auto Theft', 'Robbery', 'Harassment', 'Confidence Games', 'License Violation', 'Commercial Burglary', 'Auto Theft Recovery']
about_auto = ['Motor Vehicle Accident Response', 'Towed', 'Larceny From Motor Vehicle', 'Auto Theft', 'Auto Theft Recovery']
exist_sinner = ['Larceny', 'Drug Violation', 'Simple Assault', 'Vandalism', 'Verbal Disputes', 'Towed', 'Larceny From Motor Vehicle', 'Warrant Arrests', 'Aggravated Assault', 'Violations', 'Fraud', 'Residential Burglary', 'Auto Theft', 'Robbery', 'Harassment', 'Confidence Games', 'Disorderly Conduct', 'Firearm Violations', 'License Violation', 'Restraining Order Violations', 'Counterfeiting', 'Commercial Burglary', 'Auto Theft Recovery', 'Liquor Violation']

def func_in_list(x, lst):
    if x in lst:
        return 1
    else:
        return 0

df['HUMAN_LOSS'] = df.apply(lambda x: func_in_list(x['OFFENSE_CODE_GROUP'], human_loss), axis=1)
df['FINANCIAL_LOSS'] = df.apply(lambda x: func_in_list(x['OFFENSE_CODE_GROUP'], financial_loss), axis=1)
df['INSIDE'] = df.apply(lambda x: func_in_list(x['OFFENSE_CODE_GROUP'], inside), axis=1)
df['EXIST_VICTIM_OFFENDER'] = df.apply(lambda x: func_in_list(x['OFFENSE_CODE_GROUP'], exist_victim_offender), axis=1)
df['ABOUT_AUTO'] = df.apply(lambda x: func_in_list(x['OFFENSE_CODE_GROUP'], about_auto), axis=1)
df['EXIST_SINNER'] = df.apply(lambda x: func_in_list(x['OFFENSE_CODE_GROUP'], exist_sinner), axis=1)

del df['OFFENSE_CODE_GROUP']

df.to_csv('fixed_crime6.csv', index=False)




import pandas as pd

test_set_df = pd.read_csv('test_set.csv', engine='python')
training_set_df = pd.read_csv('training_set.csv', engine='python')
validation_set_df = pd.read_csv('validation_set.csv', engine='python')

test_set_df = pd.get_dummies(test_set_df, columns=test_set_df.columns)
training_set_df = pd.get_dummies(training_set_df, columns=training_set_df.columns)
validation_set_df = pd.get_dummies(validation_set_df, columns=validation_set_df.columns)

test_set_df.to_csv('test_set_one_hot.csv', index=False)
training_set_df.to_csv('training_set_one_hot.csv', index=False)
validation_set_df.to_csv('validation_set_one_hot.csv', index=False)



# 으악

#from ast import literal_eval

# df = df[df['Location'] != '(0.00000000, 0.00000000)']
# df = df[df['Location'] != '(-1.00000000, -1.00000000)']

# print(df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False))

# df['DISTRICT'].value_counts(sort=True, dropna=False).plot(kind='barh')
# plt.show()

# print(df.groupby('OFFENSE_CODE_GROUP'))
# print('\n'.join(df['OFFENSE_CODE_GROUP'].unique()))
# plt.show()




# transposed = np.array([literal_eval(i) for i in df['Location']]).T

# for i in transposed[0]:
#     if i < 10:
#         print(i)

# plt.scatter(transposed[0], transposed[1])
# plt.show()

# 자동차 사고 대응
# 절도죄
# 의료 지원
# 사람 조사
# 단순 폭행
# 기물 파손
# 구두 분쟁
# 견인
# 부동산 조사
# 자동차에서 폭행
