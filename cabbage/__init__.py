from cabbage.trainer import CabbageTrainer
from pandas.io.parsers import read_csv

if __name__ == '__main__':
    data = read_csv('price_data.csv', sep=',')
    t = CabbageTrainer()
    t.test(data)
    