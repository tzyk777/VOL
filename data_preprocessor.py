import numpy as np
from df_map import TimeSeriesDataFrameMap


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
        self._replace_outliers(df)
        self._resample(df)
        self._get_returns(df)

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
        df = df.resample(self.frequency).last().dropna()

    def _get_returns(self, df):
        df[TimeSeriesDataFrameMap.Returns] = (df[TimeSeriesDataFrameMap.Price] /
                                                   df[TimeSeriesDataFrameMap.Price].shift(1)).apply(np.log)
        df.drop([TimeSeriesDataFrameMap.Price], axis=1, inplace=True)
        self._replace_na(df)
