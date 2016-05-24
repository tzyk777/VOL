from abc import ABCMeta, abstractmethod
from arch import arch_model
import numpy as np


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
        res = model.fit(update_freq=100)
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



