import datetime as dt
import math
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from arch import arch_model
from dateutil.relativedelta import relativedelta

from utilities import TimeSeriesDataFrameMap, get_timestamps, min_sample_size


class VolatilityModel:
    """
    Volatility model
    Train model and make prediction
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def train_model(self, df):
        """
        :param df: pandas.DataFrame
        """
        pass


class Garch11(VolatilityModel):
    """
    Garch(1,1) model
    """
    def __init__(self, mean_model, vol_model, lags=1, dist='Normal'):
        """
        :param mean_model: str
        :param vol_model: str
        :param lags: int
        :param dist: str
        """
        self.mean_model = mean_model
        self.vol_model = vol_model
        self.lags = lags
        self.dist = dist
        self.p = 1
        self.q = 1
        # address small value, non convergence issue
        self.scaling_factor = 1000

    def train_model(self, df):
        """
        :param df: pandas.DataFrame
        :return: ARCHModelResult
        """
        model = arch_model(df * self.scaling_factor, mean=self.mean_model, lags=self.lags,
                           vol=self.vol_model, p=self.p, q=self.q, dist=self.dist)
        res = model.fit(update_freq=0, disp='off')
        return res

    def vol_forecast(self, res, steps=1):
        """
        :param res: ARCHModelResult
        :param steps: int
        :return: np.array
        """
        omega, alpha, beta = res.params['omega'], res.params['alpha[1]'], res.params['beta[1]']
        resid, cond_vol = res.resid[-1], res.conditional_volatility[-1]
        one_step = np.array(omega + alpha * resid**2 + beta * cond_vol**2)
        if steps == 1:
            return np.sqrt(one_step)/self.scaling_factor
        func = np.vectorize(lambda h: omega * (1 - (alpha + beta)**(h - 1)) / (1 - alpha - beta) + (alpha + beta)**(h - 1) * one_step)

        array = np.arange(2, steps+1)
        result_array = func(array)
        result_array = np.insert(result_array, 0, one_step)
        return np.sqrt(result_array)/self.scaling_factor

    def get_predictions(self, df, sample_size, frequency):
        """
        :param df: pandas.DataFrame
        :param sample_size: int
        :param frequency: FrequencyMap
        :return: pandas.DataFrame
        """

        months = sorted(set([dt.date(d.year, d.month, 1) for d in df.index]))[-sample_size]
        start_timestamp = df.index[-1] + dt.timedelta(days=1)
        start_timestamp = dt.datetime(start_timestamp.year, start_timestamp.month, start_timestamp.day)
        end_timestamp = start_timestamp + relativedelta(months=1)
        timestamps = list(get_timestamps(start_timestamp, end_timestamp, frequency))
        if len(df[TimeSeriesDataFrameMap.Returns]) <= min_sample_size:
            return pd.DataFrame(pd.Series.rolling(df[TimeSeriesDataFrameMap.Returns],
                                                  window=len(df[TimeSeriesDataFrameMap.Returns])).std()[-1],
                                index=timestamps,
                                columns=[TimeSeriesDataFrameMap.Volatility])
        res = self.train_model(df[months:])
        predictions = pd.DataFrame(self.vol_forecast(res, len(timestamps)), index=timestamps,
                                   columns=[TimeSeriesDataFrameMap.Volatility])
        return predictions


class RealizedVolModel:
    """
    Realized volatility model
    This class estimates realized volatility.
    Multiple models can be inherited from this class.
    For example:
        Close to Close model
        GaramanKlass
        HodgesTompkins
        Parkinson
        YangZhang
    """
    __metaclass__ = ABCMeta

    def __init__(self, df, window, clean):
        """
        :param df: pandas.DataFrame
        :param window: int
        :param clean: boolean
        """
        self.df = df
        self.window = window
        self.clean = clean

    @abstractmethod
    def get_estimator(self):
        pass


class CloseToCloseModel(RealizedVolModel):
    """
    Naive realized volatility model.
    Only consider close to close volatility.
    Close to open volatility is not considered.
    Usually this model underestimates volatility.
    """
    def __init__(self, df, window, clean):
        """
        :param df: pandas.DataFrame
        :param window: int
        :param clean: boolean
        """
        super().__init__(df, window, clean)

    def get_estimator(self):
        """
        :return: pandas.DataFrame
        """
        vol = pd.Series.rolling(self.df[TimeSeriesDataFrameMap.Returns], window=self.window).std()
        adj_factor = math.sqrt((1.0 / (1.0 - (self.window / (self.df[TimeSeriesDataFrameMap.Returns].count() - (self.window - 1.0))) + (self.window**2 - 1.0) /
                                       (3.0 * (self.df[TimeSeriesDataFrameMap.Returns].count() - (self.window - 1.0))**2))))
        result = vol * adj_factor
        result[:self.window-1] = np.nan
        result = pd.DataFrame(data=result)
        result.columns = [TimeSeriesDataFrameMap.Volatility]
        if self.clean:
            return result.dropna()
        else:
            return result


