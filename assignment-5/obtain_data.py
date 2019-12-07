import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval

df = pd.read_csv('crime.csv', engine='python')

df = df[df['Location'] != '(0.00000000, 0.00000000)']
df = df[df['Location'] != '(-1.00000000, -1.00000000)']




# transposed = np.array([literal_eval(i) for i in df['Location']]).T

# for i in transposed[0]:
#     if i < 10:
#         print(i)

# plt.scatter(transposed[0], transposed[1])
# plt.show()