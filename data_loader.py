import pandas as pd
from df_map import TimeSeriesDataFrameMap
import datetime as dt


class DataLoader:
    def __init__(self, path, start_date):
        """
        :param path:
        :param start_date
        """
        if start_date is None or start_date == '':
            raise ValueError('Start date required')

        self.path = path
        self.start_date = start_date
        self.df = pd.DataFrame()

    def load(self):
        """
        """
        self.df = pd.read_csv(self.path, sep=',', header=0)
        self.df[TimeSeriesDataFrameMap.SnapthosTime] = self.df.apply(
            lambda row: dt.datetime.combine(self.start_date + dt.timedelta(days=row['day']-1),
                                            dt.datetime.strptime(row['timestr'], '%H:%M:%S').time()), axis=1)
        self.df = self.df.set_index(pd.DatetimeIndex(self.df[TimeSeriesDataFrameMap.SnapthosTime]))
        self.df.drop(['day', 'timestr', TimeSeriesDataFrameMap.SnapthosTime], axis=1, inplace=True)

    def fetch(self, symbol, start_date, end_date):
        """
        :param symbol:
        :param start_date:
        :param end_date:
        :return:
        """
        if symbol is None or symbol == '':
            raise ValueError('Symbol required')
        if symbol not in self.df.columns:
            raise ValueError('Unknown symbol')
        if start_date is None or start_date == '':
            raise ValueError('Start date required')
        if end_date is None or end_date == '':
            raise ValueError('End date required')
        if start_date > end_date:
            raise ValueError('Start date must be later than end date')
        interested_df = self.df.loc[start_date:end_date, [symbol]]
        if interested_df.empty:
            raise ValueError('Empty dataframe')
        interested_df.rename(columns={symbol: TimeSeriesDataFrameMap.Price}, inplace=True)
        return interested_df
