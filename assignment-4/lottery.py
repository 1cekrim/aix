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

    def make_csv(self, name: str):
        self.dataframe.to_csv(name, mode='w')

def main():
    lottery = Lottery(sys.argv[1])
    lst = lottery.count_number(
        ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'bonus'])
    
    for number, count in sorted(lst, key=lambda t: t[1], reverse=True):
        print(f'{number} -> {count} times')

    lottery.add_column('win', [rnd.choice([0, 1])
                               for _ in range(len(lottery.dataframe))])
    lottery.add_column('weather', [rnd.choice(
        ['clear', 'cloudy', 'warm', 'cool']) for _ in range(len(lottery.dataframe))])

    lottery.make_csv(sys.argv[1].split('.')[0] + '-result.csv')


if __name__ == "__main__":
    main()

