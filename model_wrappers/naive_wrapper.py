import numpy as np
from model_wrappers.wrapper_abstract import ModelWrapper


class NaiveForecastWrapper(ModelWrapper):

    def __init__(self, **model_kwargs):
        self.__model_kwargs = model_kwargs
        self.__model = None
        self.__fit = None

    def fit(self, data, **data_kwargs):
        is_seasonal = self.__model_kwargs['is_seasonal']
        if is_seasonal:
            seasonal_periods = data_kwargs.get('seasonal_periods', None)
            if seasonal_periods is None:
                raise Exception('seasonal period must be provided if is_seasonal is set True')
            self.__model_kwargs['last_season'] = data[-seasonal_periods:]

        else:
            self.__model_kwargs['last_value'] = data[-1]

    def forecast(self, **forecast_kwargs):

        is_seasonal = self.__model_kwargs['is_seasonal']
        steps = forecast_kwargs.get('steps', 1)
        if is_seasonal:
            last_season = self.__model_kwargs.get('last_season', None)
            if last_season is None:
                raise Exception('Model needs to be fitted before prediction')
            reps = steps // len(last_season) + 1
            return np.tile(last_season, reps)[:steps]

        else:
            last_value = self.__model_kwargs.get('last_value', None)
            if last_value is None:
                raise Exception('Model needs to be fitted before prediction')
            return last_value * np.ones(steps)

    def to_string(self):
        return ('Naive Forecast \n'
                f'\t{self.__model_kwargs} \n')