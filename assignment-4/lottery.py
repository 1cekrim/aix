import pandas as pd
import random as rnd
import sys
from functools import reduce


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


def main():
    lottery = Lottery(sys.argv[1])

    lottery.add_column('win', [1
                               for _ in range(len(lottery.dataframe))])

    lottery.add_fake_data()

    lottery.print_dataframe()


if __name__ == "__main__":
    main()
