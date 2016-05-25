import datetime as dt
import pandas as pd
from utilities import TimeSeriesDataFrameMap

from data_loader import DataLoader, DataPreProcessor
from data_analyzer import DataAnalyzer, ErrorEstimator, VolatilityEstimator
from models import Garch11
from utilities import FrequencyMap


class Task:
    def __init__(self, path, start_date, interested_symbols, interested_start_date, interested_end_date,
                 num_std=3, frequency='1Min', forward=True, model_type='CloseToClose', clean=True):
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
        :param clean:
        """
        self.interested_symbols = interested_symbols
        self.interested_start_date = interested_start_date
        self.interested_end_date = interested_end_date
        self.frequency = frequency
        self.analysis_window = 30 if frequency is not FrequencyMap.Month else 10
        self.loader = DataLoader(path, start_date)
        self.pre_process = DataPreProcessor(num_std, frequency, forward)
        self.estimator = VolatilityEstimator(model_type, clean, frequency)
        self.data_analyzer = DataAnalyzer()
        self.model = Garch11('Constant', 'Garch')
        self.error_estimator = ErrorEstimator(self.model, self.estimator, self.frequency)

    def execute(self):
        self.loader.load()
        output = pd.DataFrame()
        for symbol in self.interested_symbols:
            df = self.loader.fetch(symbol, self.interested_start_date, self.interested_end_date)
            df = self.pre_process.pre_process(df)
            self.data_analyzer.analyze_data(df.copy())
            self.estimator.analyze_realized_vol(df, self.interested_start_date, self.interested_end_date, self.analysis_window)
            sample_size, error = self.error_estimator.get_best_sample_size(df)
            predictions = self.model.get_predictions(df, sample_size, self.frequency)
            output[symbol] = predictions[TimeSeriesDataFrameMap.Volatility]
            index = predictions.index
        output.set_index(index)
        output.to_csv(r'D:\programming\VOL\predictions.csv')


def main():
    task = Task(r'D:\programming\VOL\stockdata2.csv', dt.date(2007, 1, 1), ['a', 'b', 'c'], dt.date(2007, 1, 2), dt.date(2008, 1, 1), frequency=FrequencyMap.Hour)
    task.execute()

if __name__ == '__main__':
    main()