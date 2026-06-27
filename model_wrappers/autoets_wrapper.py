import numpy as np
import pandas as pd
from sktime.forecasting.ets import AutoETS
from model_wrappers.wrapper_abstract import ModelWrapper


class AutoETSWrapper(ModelWrapper):

    def __init__(self, **model_kwargs):
        self.__model_kwargs = model_kwargs
        self.__model = None

    def fit(self, data, **data_kwargs):
        # 'seasonal_periods' from your cross-validation call maps to sktime's 'sp'
        period = data_kwargs.pop('seasonal_periods', None)

        if period is not None:
            self.__model_kwargs['sp'] = period

        self.__model = AutoETS(**self.__model_kwargs)

        # sktime requires a pd.Series with a DatetimeIndex or RangeIndex
        series = pd.Series(data.flatten(), index=pd.RangeIndex(len(data)))
        self.__model.fit(series)

    def forecast(self, **forecast_kwargs):
        if self.__model is None:
            raise Exception('Model needs to be fitted before prediction')
        steps = forecast_kwargs.get('steps')
        fh = list(range(1, steps + 1))
        prediction = self.__model.predict(fh=fh)
        return prediction.values

    def to_string(self):
        return f'AutoETS \n\t{self.__model_kwargs}'