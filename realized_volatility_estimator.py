from abc import ABCMeta, abstractmethod
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

from df_map import TimeSeriesDataFrameMap, VolatilityModelsMap


class RealizedVolModel:
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


class CloseToCloseModel(RealizedVolModel):
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
        vol = pd.rolling_std(self.df[TimeSeriesDataFrameMap.Returns], window=self.window)
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
    def __init__(self, model_type, clean, frequency):
        """
        :param model_type:
        :param clean:
        :param frequency:
        """
        self.model_type = model_type
        self.clean = clean
        self.frequency = frequency

        if self.model_type is None or self.model_type == '':
            raise ValueError('Model type required')

        self.model_type = self.model_type.lower()

        if self.model_type not in [VolatilityModelsMap.CloseToClose]:
            raise ValueError('Acceptable realized_volatility model is required')
        
    def get_estimator(self, df, window):
        """
        :param df
        :param window
        :return:
        """
        if len(df) <= window:
            raise ValueError('Dataset is too small {size} compared to rolling windows {window}'.format(
                size=len(df),
                window=window
            ))

        if self.model_type == VolatilityModelsMap.CloseToClose:
            return CloseToCloseModel(df, window, self.clean).get_estimator()

    def analyze_realized_vol(self, df, interested_start_date, interested_end_date, window):
        """
        :param df:
        :param interested_start_date:
        :param interested_end_date:
        :param window:
        """
        vol = self.get_estimator(df, window)
        if self.frequency == '1Min':
            groups = [vol.index.hour, vol.index.minute]
        elif self.frequency == 'H':
            groups = [vol.index.hour]
        elif self.frequency == 'D':
            groups = [vol.index.day]
        elif self.frequency == 'M':
            groups = [vol.index.month]
        else:
            raise ValueError('Unknown frequency {frequency}'.format(frequency=self.frequency))

        title, xlabel = self._get_documents()
        agg_minute = vol.groupby(groups).mean()
        agg_plt = agg_minute[TimeSeriesDataFrameMap.Volatility].plot(
            title=title.format(
            start_date=interested_start_date,
            end_date=interested_end_date))

        agg_plt.set_xlabel(xlabel)
        agg_plt.set_ylabel('Realized Volatility %')
        plt.show()

    def _get_documents(self):
        """
        :return:
        """
        if self.frequency == '1Min':
            return 'Average intraday minute realized volatility between {start_date} and {end_date}', 'Hour-Minute'
        elif self.frequency == 'H':
            return 'Average intraday hourly realized volatility between {start_date} and {end_date}', 'Hour'
        elif self.frequency == 'D':
            return 'Average daily realized volatility between {start_date} and {end_date}', 'Day'
        elif self.frequency == 'M':
            return 'Average monthly realized volatility between {start_date} and {end_date}', 'Month'
