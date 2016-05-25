import pandas as pd
from utilities import TimeSeriesDataFrameMap
import datetime as dt
import numpy as np


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


class DataPreProcessor:
    def __init__(self, num_std=3, frequency='1Min', forward=True):
        """
        :param num_std:
        :param frequency
        :param forward:
        """
        self.num_std = num_std
        self.frequency = frequency
        self.forward = forward

    def pre_process(self, df):
        """
        :param df:
        """
        self._replace_outliers(df)
        df = self._resample(df)
        self._get_returns(df)
        return df

    def _replace_outliers(self, df):
        df[np.abs(df - df.mean()) > self.num_std * df.std()] = np.nan

    def _replace_na(self, df):
        if self.forward:
            df.fillna(method='pad', inplace=True)
            df.fillna(method='bfill', inplace=True)
        else:
            df.fillna(method='bfill', inplace=True)
            df.fillna(method='pad', inplace=True)

    def _resample(self, df):
        return df.resample(self.frequency).last().dropna()

    def _get_returns(self, df):
        df[TimeSeriesDataFrameMap.Returns] = (df[TimeSeriesDataFrameMap.Price] /
                                                   df[TimeSeriesDataFrameMap.Price].shift(1)).apply(np.log)
        df.drop([TimeSeriesDataFrameMap.Price], axis=1, inplace=True)
        self._replace_na(df)
