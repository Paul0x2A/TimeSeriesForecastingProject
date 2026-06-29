from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from model_wrappers.wrapper_abstract import ModelWrapper


class ExponentialSmoothingWrapper(ModelWrapper):

    def __init__(self, **model_kwargs):
        self.__model = None
        self.__fit = None
        defaults = {
            'error': 'add',
            'trend': None,
            'damped_trend': False,
            'seasonal': None,
            'seasonal_periods': None,
            'initialization_method': 'estimated',
            'initial_level': None,
            'initial_trend': None,
            'initial_seasonal': None,
            'bounds': None,
            'dates': None,
            'freq': None,
            'missing': 'none'
        }
        self.__model_kwargs = {**defaults, **model_kwargs}

    def fit(self, data, **data_kwargs):
        self.__model_kwargs['seasonal_periods'] = data_kwargs.get('seasonal_periods', None)
        self.__model = ETSModel(data, **self.__model_kwargs)
        self.__fit = self.__model.fit()

    def forecast(self, **forecast_kwargs):
        if self.__model is None:
            raise Exception('Model needs to be fitted before prediction')
        return self.__fit.forecast(**forecast_kwargs)

    def to_string(self):
        return ('Exponential Smoothing \n'
                f'\t{self.__model_kwargs} \n')