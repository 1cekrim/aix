import pandas as pd
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import sys
from functools import reduce
from sklearn.cluster import KMeans


class Lottery:
    def __init__(self, file_name: str):
        self.dataframe = pd.read_csv(file_name)

    def _make_lst(self, column_names: list) -> list:
        return reduce(lambda a, b: a + b, map(lambda i: list(self.dataframe[i]), column_names))

    def count_number(self, column_names: list, min: int = 1, max: int = 45) -> list:
        lst = self._make_lst(column_names)
        return map(lambda i: (i, lst.count(i)), range(min, max + 1))

    def add_column(self, column_name: str, data: list):
        self.dataframe[column_name] = data

    def add_fake_data(self):
        s = [sorted(i) for i in map(lambda j: j[2:9],
                                    self.dataframe.values.tolist())]
        result = []
        for data in self.dataframe.values.tolist():
            while True:
                numbers = rnd.sample([i for i in range(1, 46)], 7)
                sorted_numbers = sorted(numbers)
                if not reduce(lambda a, b: a or b, [k == sorted_numbers for k in s], False):
                    result.append(data[0:2] + numbers + [0])
                    break
        self.dataframe = self.dataframe.append(pd.DataFrame(
            result, columns=self.dataframe.columns)).sort_values(by=['round', 'win'], ascending=False)

    def print_dataframe(self):
        print(self.dataframe[0:20])

    def make_csv(self, name: str):
        self.dataframe.to_csv(name, mode='w')


class AnalysisLottery:
    def __init__(self, lottery: Lottery):
        self.lottery = lottery
        self.data = []

    def map(self, func) -> list:
        result = []
        for data in self.lottery.dataframe.values.tolist():
            result.append(func(data))
        return result

    def append(self, lst: list):
        self.data.append(lst)

    def get_data(self):
        return self.data

    def print(self):
        print(self.data)

    def kmean_plot(self):
        transposed = np.array(self.data).T
        model = KMeans(n_clusters=2).fit(transposed)
        predict = model.fit_predict(transposed)
        plt.scatter(self.data[0], self.data[1],
                    c=predict, s=50, cmap='viridis')
        plt.show()

    def scatter_plot(self, win: list):
        plt.scatter(self.data[0], self.data[1],
                    c=win, s=50, cmap='viridis')
        plt.show()


class AnalysisFunctions:
    @classmethod
    def mean(cls):
        return lambda lst: reduce(lambda a, b: a + b, cls.parse_numbers(lst), 0.0) / len(cls.parse_numbers(lst))

    @classmethod
    def min(cls):
        return lambda lst: reduce(lambda a, b: min(a, b), cls.parse_numbers(lst))

    @classmethod
    def max(cls):
        return lambda lst: reduce(lambda a, b: max(a, b), cls.parse_numbers(lst))

    @classmethod
    def variance(cls, mean: float):
        return lambda lst: reduce(lambda a, b: a + (mean - b)**2, cls.parse_numbers(lst), 0.0) / len(cls.parse_numbers(lst))

    @classmethod
    def standard_variance(cls, mean: float):
        return lambda lst: cls.variance(mean)(lst)**0.5

    @classmethod
    def parse_numbers(cls, lst: list) -> list:
        return lst[2:9]

    @classmethod
    def parse_date(cls, lst: list) -> list:
        return lst[1].split('.')


def main():
    lottery = Lottery(sys.argv[1])
    lottery.add_column('win', [1
                               for _ in range(len(lottery.dataframe))])

    lst = lottery.count_number(
        ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'bonus'])
    lst = sorted(lst, key=lambda t: t[1])

    print('prediction for next week’s winning number:', end=' ')
    for i in range(7):
        print(lst[i][0], end=' ')
    print('\n\n')

    lottery.add_fake_data()
    lottery.print_dataframe()
    analysis = AnalysisLottery(lottery)

    analysis.append(analysis.map(AnalysisFunctions.mean()))
    analysis.append(analysis.map(lambda lst: AnalysisFunctions.standard_variance(
        int(AnalysisFunctions.parse_date(lst)[2]) + int(AnalysisFunctions.parse_date(lst)[1]))(lst)))

    # analysis.kmean_plot()
    # analysis.scatter_plot(lottery.dataframe['win'])

    lottery.make_csv('result.csv')


if __name__ == "__main__":
    main()
