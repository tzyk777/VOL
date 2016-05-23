from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt

from df_map import TimeSeriesDataFrameMap


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

        plot_acf(df[TimeSeriesDataFrameMap.Residuals], axes[1], lags=12)
        label(axes[1], 'Residuals autocorrelation')

        plot_acf(df[TimeSeriesDataFrameMap.Abs_residuals], axes[2], lags=12)
        label(axes[2], 'Absolute residuals autocorrelation')

        plot_acf(df[TimeSeriesDataFrameMap.Square_residuals], axes[3], lags=12)
        label(axes[3], 'Square residuals autocorrelation')

        plot_pacf(df[TimeSeriesDataFrameMap.Square_residuals], axes[4], lags=12)
        label(axes[4], 'Square residuals partial autocorrelation')
        plt.show()

    @staticmethod
    def test_autocorr(df):
        """
        :param df:
        """
        lbvalue, pvalue, bpvalue, bppvalue = acorr_ljungbox(df[TimeSeriesDataFrameMap.Square_residuals], lags=12, boxpierce=True)
        print('Ljung Box Test')
        print('Lag  P-value')
        for l, p in zip(range(1, 13), pvalue):
            print(l, ' ', p)


