import datetime as dt

import numpy as np
import pandas as pd

from utilities import TimeSeriesDataFrameMap, FrequencyMap


class DataLoader:
    """
    Raw data loader
    """
    def __init__(self, path, start_date):
        """
        :param path: str
        :param start_date: datetime.date
        """
        if start_date is None or start_date == '':
            raise ValueError('Start date required')

        self.path = path
        self.start_date = start_date
        self.df = pd.DataFrame()

    def load(self):
        self.df = pd.read_csv(self.path, sep=',', header=0)
        self.df[TimeSeriesDataFrameMap.SnapthosTime] = self.df.apply(
            lambda row: dt.datetime.combine(self.start_date + dt.timedelta(days=row['day']-1),
                                            dt.datetime.strptime(row['timestr'], '%H:%M:%S').time()), axis=1)
        self.df = self.df.set_index(pd.DatetimeIndex(self.df[TimeSeriesDataFrameMap.SnapthosTime]))
        self.df.drop(['day', 'timestr', TimeSeriesDataFrameMap.SnapthosTime], axis=1, inplace=True)

    def fetch(self, symbol, start_date, end_date):
        """
        :param symbol: str
        :param start_date: datetime.date
        :param end_date: datetime.date
        :return: pandas.DataFrame
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
    """
    Data pre process class
    Fill NA value by using different strategy
    Identify and exclude outliers
    """
    def __init__(self, num_std=3, frequency=FrequencyMap.Minute, forward=True):
        """
        :param num_std: int
        :param frequency: FrequencyMap
        :param forward: boolean
        """
        self.num_std = num_std
        self.frequency = frequency
        self.forward = forward

    def pre_process(self, df):
        """
        :param df: pandas.DataFrame
        :return: pandas.DataFrame
        """
        self._replace_outliers(df)
        df = self._resample(df)
        self._get_returns(df)
        df[TimeSeriesDataFrameMap.Returns] = df[TimeSeriesDataFrameMap.Returns]
        return df

    def _replace_outliers(self, df):
        """
        :param df: pandas.DataFrame
        """
        df[np.abs(df - df.mean()) > self.num_std * df.std()] = np.nan

    def _replace_na(self, df):
        """
        :param df: pandas.DataFrame
        """
        if self.forward:
            df.fillna(method='pad', inplace=True)
            df.fillna(method='bfill', inplace=True)
        else:
            df.fillna(method='bfill', inplace=True)
            df.fillna(method='pad', inplace=True)

    def _resample(self, df):
        """
        :param df: pandas.DataFrame
        :return: pandas.DataFrame
        """
        return df.resample(self.frequency).last().dropna()

    def _get_returns(self, df):
        """
        :param df: pandas.DataFrame
        :return: pandas.DataFrame
        """
        df[TimeSeriesDataFrameMap.Returns] = (df[TimeSeriesDataFrameMap.Price] /
                                                   df[TimeSeriesDataFrameMap.Price].shift(1)).apply(np.log)
        df.drop([TimeSeriesDataFrameMap.Price], axis=1, inplace=True)
        self._replace_na(df)
