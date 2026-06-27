from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
from model_wrappers.wrapper_abstract import ModelWrapper


class ExponentialSmoothingWrapper(ModelWrapper):

    def __init__(self, **model_kwargs):
        self.__model_kwargs = model_kwargs
        self.__model = None
        self.__fit = None

    def fit(self, data, **data_kwargs):
        self.__model_kwargs = self.__model_kwargs | data_kwargs
        self.__model = ExponentialSmoothing(data, **self.__model_kwargs)
        self.__fit = self.__model.fit(optimized=True)

    def forecast(self, **forecast_kwargs):
        if self.__model is None:
            raise Exception('Model needs to be fitted before prediction')
        return self.__fit.forecast(**forecast_kwargs)

    def to_string(self):
        return ('Exponential Smoothing \n'
                f'\t{self.__model_kwargs} \n')