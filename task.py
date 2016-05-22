import datetime as dt
import matplotlib.pyplot as plt

from data_loader import DataLoader
from data_preprocessor import DataPreProcessor
from realized_volatility_estimator import VolatilityEstimator
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
        self.estimator = VolatilityEstimator(model_type, window, clean)

    def execute(self):
        self.loader.load()
        for symbol in self.interested_symbols:
            df = self.loader.fetch(symbol, self.interested_start_date, self.interested_end_date)
            self.pre_process.pre_process(df)
            vol = self.estimator.get_estimator(df)
            agg_minute = vol.groupby([vol.index.hour, vol.index.minute]).mean()
            agg_plt = agg_minute[TimeSeriesDataFrameMap.Volatility].plot(
                title='Average intraday realized volatility between {start_date} and {end_date}'.format(
                start_date=self.interested_start_date,
                end_date=self.interested_end_date))

            agg_plt.set_xlabel('Hour-Minute')
            agg_plt.set_ylabel('Realized Volatility %')
            plt.show()


def main():
    task = Task(r'D:\programming\VOL\stockdata2.csv', dt.date(2007, 1, 1), ['a', 'b', 'c', 'd', 'e', 'f'], dt.date(2007, 1, 2), dt.date(2008, 1, 1))
    task.execute()

if __name__ == '__main__':
    main()