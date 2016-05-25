from abc import ABCMeta, abstractmethod
from arch import arch_model
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import pandas as pd

from utilities import TimeSeriesDataFrameMap, get_timestamps


class VolatilityModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train_model(self, df):
        pass


class Garch11(VolatilityModel):
    def __init__(self, mean_model, vol_model, lags=1, dist='Normal'):
        """
        :param mean_model:
        :param vol_model:
        :param lags:
        :param p:
        :param q:
        :param dist:
        """
        self.mean_model = mean_model
        self.vol_model = vol_model
        self.lags = lags
        self.p = 1
        self.q = 1
        self.dist = dist

    def train_model(self, df):
        """
        :param df:
        :param last_obs:
        :return:
        """
        model = arch_model(df, mean=self.mean_model, lags=self.lags,
                           vol=self.vol_model, p=self.p, q=self.q, dist=self.dist)
        res = model.fit(update_freq=0, disp='off')
        return res

    @staticmethod
    def vol_forecast(res, steps=1):
        """
        :param res:
        :param steps:
        :return:
        """
        omega, alpha, beta = res.params['omega'], res.params['alpha[1]'], res.params['beta[1]']
        resid, cond_vol = res.resid[-1], res.conditional_volatility[-1]
        one_step = np.array(omega + alpha * resid**2 + beta * cond_vol**2)
        if steps == 1:
            return np.sqrt(one_step)
        func = np.vectorize(lambda h: omega * (1 - (alpha + beta)**(h - 1)) / (1 - alpha - beta) + (alpha + beta)**(h - 1) * one_step)

        array = np.arange(2, steps+1)
        result_array = func(array)
        result_array = np.insert(result_array, 0, one_step)
        return np.sqrt(result_array)

    def get_predictions(self, df, sample_size, frequency):
        """
        :param df:
        :param sample_size:
        :param frequency:
        :return:
        """
        months = sorted(set([dt.date(d.year, d.month, 1) for d in df.index]))[-sample_size]
        res = self.train_model(df[months:])
        start_timestamp = df.index[-1] + dt.timedelta(days=1)
        start_timestamp = dt.datetime(start_timestamp.year, start_timestamp.month, start_timestamp.day)
        end_timestamp = start_timestamp + relativedelta(months=1)
        timestamps = list(get_timestamps(start_timestamp, end_timestamp, frequency))
        predictions = pd.DataFrame(self.vol_forecast(res, len(timestamps)), index=timestamps,
                                   columns=[TimeSeriesDataFrameMap.Volatility])
        return predictions



