import datetime as dt

from dateutil.relativedelta import relativedelta

min_sample_size = 20


class TimeSeriesDataFrameMap:
    Symbol = 'symbol'
    SnapthosTime = 'time'
    Price = 'price'
    Returns = 'returns'
    Volatility = 'volatility'
    Square_residuals = 'square_residuals'
    Abs_residuals = 'abs_residuals'
    Residuals = 'residuals'
    Cond_volatility = 'cond_volatility'
    Error = 'error'


class VolatilityModelsMap:
    CloseToClose = 'closetoclose'


class FrequencyMap:
    Minute = '1Min'
    Hour = 'H'
    Day = 'D'
    Month = 'M'


def get_timestamps(start_timestamp, end_timestamp, frequency):
    """
    :param start_timestamp: datetime.datetime
    :param end_timestamp: datetime.datetime
    :param frequency: FrequencyMap
    :return: Generator
    """
    current_timestamp = start_timestamp
    if frequency == FrequencyMap.Minute:
        delta = dt.timedelta(minutes=1)
    elif frequency == FrequencyMap.Hour:
        delta = dt.timedelta(hours=1)
    elif frequency == FrequencyMap.Day:
        delta = dt.timedelta(days=1)
    elif frequency == FrequencyMap.Month:
        delta = relativedelta(months=1)
    else:
        raise ValueError('Unknown frequency {frequency}'.format(frequency=frequency))

    while current_timestamp < end_timestamp:
        if current_timestamp.isoweekday() in range(1, 6):
            if frequency not in [FrequencyMap.Minute, FrequencyMap.Hour] or current_timestamp.hour in range(9, 17):
                yield current_timestamp
        current_timestamp += delta
