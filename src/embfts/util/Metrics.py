import pandas as pd
from pyFTS.common import Util
from pyFTS.benchmarks import Measures
import math
import datetime

class Metrics():
    def __init__(self):
        self.name = 'Accuracy Metrics'
        self.shortname = 'metrics'

    def nrmse(self, rmse, y):
        x = max(y) - min(y)
        return (rmse / x)



