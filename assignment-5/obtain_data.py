# import pandas as pd

# df = pd.read_csv('crime.csv', engine='python')
# df = df.drop(columns=['INCIDENT_NUMBER', 'OFFENSE_CODE', 'OFFENSE_DESCRIPTION', 'Lat', 'Long', 'Location', 'STREET', 'REPORTING_AREA', 'SHOOTING', 'UCR_PART', 'OCCURRED_ON_DATE'])
# df.to_csv('fixed_crime.csv', index=False)



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

df.to_csv('fixed_crime6.csv', index=False)


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
