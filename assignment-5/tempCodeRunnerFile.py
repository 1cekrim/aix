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