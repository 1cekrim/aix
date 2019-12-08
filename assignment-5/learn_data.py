import pandas as pd
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

target_column = ['DISTRICT', 'YEAR', 'MONTH', 'DAY_OF_WEEK', 'HOUR', 'CLASSIFICATION', 'TIME', 'WEEKDAY']

df = pd.read_csv('fixed_crime6.csv', engine='python')

one_hot_index = {}

cnt = 0

for i in target_column:
    one_hot_index[i] = {}
    for c in df[i].unique():
        one_hot_index[i][c] = cnt
        cnt += 1

print(one_hot_index)