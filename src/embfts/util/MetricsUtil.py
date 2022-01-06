import numpy as np
from pyFTS.benchmarks import Measures
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

class MetricsUtil():
    def __init__(self):
        self.name = 'Accuracy Metrics'
        self.shortname = 'metrics'

    def nrmse(self, rmse, y):
        x = max(y) - min(y)
        return (rmse / x)

    def compute_all_metrics(self,target,forecast, results):

        rmse = Measures.rmse(target, forecast)
        mape = Measures.mape(target, forecast)
        smape = Measures.smape(target, forecast)
        mae = mean_absolute_error(target, forecast)
        r2 = r2_score(target, forecast)
        nrmse = Measures.nmrse(target, forecast) if not np.isinf(Measures.nmrse(target, forecast)) else 0

        results["rmse"].append(rmse)
        results["mape"].append(mape)
        results["smape"].append(smape)
        results["mae"].append(mae)
        results["r2"].append(r2)
        results["nrmse"].append(nrmse)

        return results



