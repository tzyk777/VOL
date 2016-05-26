import datetime as dt

import pandas as pd

from data_analyzer import DataAnalyzer, ErrorEstimator, VolatilityEstimator
from data_loader import DataLoader, DataPreProcessor
from models import Garch11
from utilities import FrequencyMap
from utilities import TimeSeriesDataFrameMap


class Task:
    """
    Core class of this volatility library
    It schedules each component.
    Read raw data.
    Pre process data.
    Analyze data.
    Make prediction.
    Output results.
    """
    def __init__(self, path, start_date, interested_symbols, interested_start_date, interested_end_date,
                 num_std=3, frequency=FrequencyMap.Minute, forward=True, model_type='CloseToClose', clean=True):
        """
        :param path: str
        :param start_date: datetime.date
        :param interested_symbols: list[str]
        :param interested_start_date: datetime.date
        :param interested_end_date: datetime.date
        :param num_std: int
        :param frequency: FrequencyMap
        :param forward: boolean
        :param model_type: str
        :param clean: boolean
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
        file_name = r'D:\programming\VOL\{frequency}_predictions.csv'.format(frequency=self.frequency)
        output.to_csv(file_name)


def main():
    task = Task(r'D:\programming\VOL\stockdata2.csv', dt.date(2007, 1, 1), ['a', 'b', 'c', 'd', 'e', 'f'], dt.date(2007, 1, 2), dt.date(2008, 1, 1), frequency=FrequencyMap.Minute)
    task.execute()

if __name__ == '__main__':
    main()