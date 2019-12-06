import pandas as pd
import re
import csv
# offense_codes.csv에 인코딩이 잘못된 문자가 있기 때문에, engine을 c에서 python으로 바꿔줘야 합니다.
df = pd.read_csv('offense_codes.csv', engine='python')
# CODE값을 기준으로 오름차순으로 정렬해줍니다.
df = df.sort_values(['CODE'])
# NAME 값에서 필요없는 문자들을 지워줍니다.
df['NAME'] = df['NAME'].map(lambda x: re.sub("[^a-zA-Z0-9-()$&.,/]", " ", x))
# 중복된 값들을 지줘줍니다.
df = df.drop_duplicates(subset='CODE', keep='first')
# fixed_offense_codes.csv에 결과물을 적어줍니다.
df.to_csv('fixed_offense_codes.csv', index=False)
