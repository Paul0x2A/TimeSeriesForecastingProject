from enum import Enum

from anaconda_navigator.api.team_edition_api import repo_utils
from sklearn.model_selection import TimeSeriesSplit

class Frequency(Enum):
    # Values = (periodicity, forecasting horizon)
    HOURLY = (24, 48)
    DAILY = (7, 14)
    WEEKLY = (52, 13)
    MONTHLY = (12, 18)
    QUARTERLY = (4, 8)
    ANNUALLY = (1, 6)

    def get_period(self):
        return self.value[0]

    def get_forecasting_horizon(self):
        return self.value[1]

    def get_tscv(self, n_splits=5, gap=0):
        # return tscv object based on forecasting horizon for each sampling frequency
        return TimeSeriesSplit(n_splits=n_splits, test_size=self.value[1], gap=gap)

