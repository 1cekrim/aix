import pandas as pd

df = pd.read_csv('fixed_crime6.csv', engine='python')

corr = df.corr(method='pearson')
print(corr)