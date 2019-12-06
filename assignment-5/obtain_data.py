import pandas as pd

df = pd.read_csv('crime.csv', engine='python')
print(df['OFFENSE_CODE_GROUP'].value_counts().iloc[:10])