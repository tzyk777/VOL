from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import datetime as dt
from dateutil.relativedelta import relativedelta
from collections import defaultdict
from abc import ABCMeta, abstractmethod
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

from utilities import TimeSeriesDataFrameMap, VolatilityModelsMap, FrequencyMap


class DataAnalyzer:

    def analyze_data(self, df):
        """
        :param df:
        """
        self.get_residuals(df)
        self.draw_ACFs(df)
        self.test_autocorr(df)

    @staticmethod
    def get_residuals(df):
        df[TimeSeriesDataFrameMap.Residuals] = df[TimeSeriesDataFrameMap.Returns] - df[TimeSeriesDataFrameMap.Returns].mean()
        df[TimeSeriesDataFrameMap.Abs_residuals] = df[TimeSeriesDataFrameMap.Residuals].abs()
        df[TimeSeriesDataFrameMap.Square_residuals] = df[TimeSeriesDataFrameMap.Residuals]**2

    @staticmethod
    def draw_ACFs(df):
        """
        :param df:
        """
        def label(ax, string):
            ax.annotate(string, (1, 1), xytext=(-8, -8), ha='right', va='top',
                        size=14, xycoords='axes fraction', textcoords='offset points')

        fig, axes = plt.subplots(nrows=5, figsize=(8, 12))
        fig.tight_layout()

        axes[0].plot(df[TimeSeriesDataFrameMap.Square_residuals])
        label(axes[0], 'Returns')

        plot_acf(df[TimeSeriesDataFrameMap.Residuals], axes[1], lags=10)
        label(axes[1], 'Residuals autocorrelation')

        plot_acf(df[TimeSeriesDataFrameMap.Abs_residuals], axes[2], lags=10)
        label(axes[2], 'Absolute residuals autocorrelation')

        plot_acf(df[TimeSeriesDataFrameMap.Square_residuals], axes[3], lags=10)
        label(axes[3], 'Square residuals autocorrelation')

        plot_pacf(df[TimeSeriesDataFrameMap.Square_residuals], axes[4], lags=10)
        label(axes[4], 'Square residuals partial autocorrelation')
        plt.show()

    @staticmethod
    def test_autocorr(df):
        """
        :param df:
        """
        lbvalue, pvalue, bpvalue, bppvalue = acorr_ljungbox(df[TimeSeriesDataFrameMap.Square_residuals], lags=10, boxpierce=True)
        print('Ljung Box Test')
        print('Lag  P-value')
        for l, p in zip(range(1, 13), pvalue):
            print(l, ' ', p)


class ErrorEstimator:
    def __init__(self, model, realized_vol_estimator, frequency):
        """
        :param model:
        :param realized_vol_estimator:
        :param frequency:
        """
        self.model = model
        self.realized_vol_estimator = realized_vol_estimator
        self.frequency = frequency

    def _get_estimated_errors(self, train_df, test_df):
        """
        :param train_df:
        :param test_df:
        :return:
        """
        param = self.model.train_model(train_df)
        predictions = self.model.vol_forecast(param, len(test_df))
        df = pd.concat([train_df, test_df])
        cond_vols = np.concatenate((np.array(param.conditional_volatility), predictions))
        df[TimeSeriesDataFrameMap.Cond_volatility] = pd.Series(cond_vols, index=df.index)
        real_vol = self.realized_vol_estimator.get_realized_vol(df, len(train_df))
        df = pd.merge(df, real_vol, left_index=True, right_index=True)
        df[TimeSeriesDataFrameMap.Error] = (df[TimeSeriesDataFrameMap.Cond_volatility] - df[TimeSeriesDataFrameMap.Volatility])**2
        return df[TimeSeriesDataFrameMap.Error].sum()

    def get_best_sample_size(self, df):
        """
        :param df:
        :return:
        """
        errors = defaultdict(list)
        months = sorted(set([dt.date(d.year, d.month, 1) for d in df.index]))
        for length in range(1, len(months)):
            current_months = months[:-length]
            for index, train_start in enumerate(current_months):
                train_end = months[index+length]
                test_start = train_end
                test_end = test_start + relativedelta(months=1)
                train_df, test_df = df[train_start: train_end], df[test_start: test_end]
                errors[length].append(self._get_estimated_errors(train_df, test_df))
        sample_size, min_error = 1, np.mean(errors[1])
        for length, err in errors.items():
            current_err = np.mean(err)
            if current_err < min_error:
                min_error = current_err
                sample_size = length
        return sample_size, min_error


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

    def get_realized_vol(self, df, window):
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
        vol = self.get_realized_vol(df, window)
        if self.frequency == FrequencyMap.Minute:
            groups = [vol.index.hour, vol.index.minute]
        elif self.frequency == FrequencyMap.Hour:
            groups = [vol.index.hour]
        elif self.frequency == FrequencyMap.Day:
            groups = [vol.index.day]
        elif self.frequency == FrequencyMap.Month:
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
