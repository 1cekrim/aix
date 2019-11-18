import pandas as pd

# lottery.csv 파일에서 dataframe을 가져옵니다
dataframe = pd.read_csv('./lottery.csv')

# data를 저장할 dict 입니다
data = {}

# dataframe에서 column들의 값을 읽어와 data에 넣습니다
for c in dataframe.columns:
    data[c] = dataframe[c]

# 숫자를 찾을 이름들입니다
names = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'bonus']

# 숫자의 갯수를 저장할 list 입니다
numbers = [0 for _ in range(0, 46)]

# 숫자를 셉니다
for name in names:
    for number in data[name]:
        numbers[number] += 1

for number in range(1, 46):
    print(f'{number}: {numbers[number]}')
