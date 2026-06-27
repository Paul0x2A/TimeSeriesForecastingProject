from abc import ABC, abstractmethod
import numpy as np

class ModelWrapper(ABC):
    @abstractmethod
    def fit(self, data, period, **fit_kwargs):
        # implement a wrapper for the method's fit function
        pass

    @abstractmethod
    def forecast(self, **forecast_kwargs)->np.ndarray:
        # implement a wrapper for the method's fit-forecast function
        pass

    @abstractmethod
    def to_string(self):
        # print information about the method and it's arguments/hyperparameters used
        pass
