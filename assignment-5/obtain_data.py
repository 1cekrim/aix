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



import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('fixed_crime2.csv', engine='python')
value_counts = df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False).tolist()
value_counts_index = df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False).index.tolist()

drop_list = []

for i in range(len(value_counts)):
    if value_counts[i] <= 1000:
        drop_list.append(value_counts_index[i])

df = df[~df['OFFENSE_CODE_GROUP'].isin(drop_list)]

df.to_csv('fixed_crime3.csv', index=False)

# df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False).plot(kind='barh')
# plt.show()

print(len(df['OFFENSE_CODE_GROUP'].value_counts(sort=True, dropna=False).index.tolist()))



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
