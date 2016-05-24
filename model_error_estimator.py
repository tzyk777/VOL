import numpy as np
import pandas as pd

from df_map import TimeSeriesDataFrameMap


class ErrorEstimator:
    def __init__(self, model, realized_vol_estimator):
        """
        :param model:
        :param realized_vol_estimator:
        """
        self.model = model
        self.realized_vol_estimator = realized_vol_estimator

    def estimate(self, train_df, test_df):
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
        real_vol = self.realized_vol_estimator.get_estimator(df, len(train_df))
        df = pd.merge(df, real_vol, left_index=True, right_index=True)


