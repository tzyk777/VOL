from abc import ABCMeta, abstractmethod
import pandas as pd
import math
import numpy as np

from df_map import TimeSeriesDataFrameMap, VolatilityModelsMap


class VolatilityModel:
    __metaclass__ = ABCMeta

    def __init__(self, df, window, clean):
        """
        :param df:
        :param window:
        :param clean:
        """
        self.df = df
        self.window = window
        self.clean = clean

    @abstractmethod
    def get_estimator(self):
        pass


class CloseToCloseModel(VolatilityModel):
    def __init__(self, df, window, clean):
        """
        :param df:
        :param window:
        :param clean:
        """
        super().__init__(df, window, clean)

    def get_estimator(self):
        """
        :return:
        """
        vol = pd.rolling_std(self.df[TimeSeriesDataFrameMap.Returns], window=self.window) * math.sqrt(252)
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


class VolatilityEstimator(object):
    def __init__(self, model_type, window, clean):
        """
        :param model_type:
        :param window:
        :param clean:
        """
        self.model_type = model_type
        self.window = window
        self.clean = clean

    def get_estimator(self, df):
        """
        :param df
        :return:
        """
        if len(df) <= self.window:
            raise ValueError('Dataset is too small {size} compared to rolling windows {windows}'.format(
                len(df),
                self.window
            ))

        if self.model_type is None or self.model_type == '':
            raise ValueError('Model type required')
        self.model_type = self.model_type.lower()
        if self.model_type not in [VolatilityModelsMap.CloseToClose]:
            raise ValueError('Acceptable realized_volatility model is required')

        if self.model_type == VolatilityModelsMap.CloseToClose:
            return CloseToCloseModel(df, self.window, self.clean).get_estimator()
