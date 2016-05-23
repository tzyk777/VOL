import datetime as dt

from data_loader import DataLoader
from data_preprocessor import DataPreProcessor
from realized_volatility_estimator import VolatilityEstimator
from data_analysis import DataAnalyzer
from df_map import TimeSeriesDataFrameMap


class Task:
    def __init__(self, path, start_date, interested_symbols, interested_start_date, interested_end_date,
                 num_std=3, frequency='1Min', forward=True, model_type='CloseToClose', window=30, clean=True):
        """
        :param path:
        :param start_date:
        :param interested_symbols:
        :param interested_start_date:
        :param interested_end_date:
        :param num_std:
        :param frequency:
        :param forward:
        :param model_type:
        :param window:
        :param clean:
        """
        self.interested_symbols = interested_symbols
        self.interested_start_date = interested_start_date
        self.interested_end_date = interested_end_date
        self.loader = DataLoader(path, start_date)
        self.pre_process = DataPreProcessor(num_std, frequency, forward)
        self.estimator = VolatilityEstimator(model_type, window, clean, frequency)
        self.data_analyzer = DataAnalyzer()

    def execute(self):
        self.loader.load()
        for symbol in self.interested_symbols:
            df = self.loader.fetch(symbol, self.interested_start_date, self.interested_end_date)
            df = self.pre_process.pre_process(df)
            self.estimator.analyze_realized_vol(df, self.interested_start_date, self.interested_end_date)
            self.data_analyzer.analyze_data(df.copy())



def main():
    task = Task(r'D:\programming\VOL\stockdata2.csv', dt.date(2007, 1, 1), ['a'], dt.date(2007, 1, 2), dt.date(2008, 1, 1), frequency='H')
    task.execute()

if __name__ == '__main__':
    main()