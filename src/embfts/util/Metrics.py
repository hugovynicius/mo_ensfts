import pandas as pd
from pyFTS.common import Util
from pyFTS.benchmarks import Measures
import math
import datetime

class Metrics():
    def __init__(self,models, testdata):
        self.models = models
        self.testdata = testdata

    def rmse_many(self):
        rows = []
        for file in self.models:
            try:
                model = Util.load_obj(file);
                row = [model.shortname, model.order, len(model)];
                rmse, _, _ = Measures.get_point_statistics(self.testdata, model);
                row.append(rmse);
                rows.append(row);
            except:
                pass
        metrics = pd.DataFrame(rows, columns=["Model", "Order", "Size", "RMSE"]).sort_values(["RMSE", "Size"]);
        return (metrics);

    # rmse, mape, u = Measures.get_point_statistics(data.values[:700], model)

    def sliding_window_statistics(self, data, model, n_windows, windows_length=None, train_percentage=0.7):
        if windows_length is None:
            windows_length = math.floor(len(data) / n_windows)
            print(windows_length)
        train_length = math.floor(windows_length * train_percentage)
        last_cut = 0

        result = {
            "window": [],
            "rmse": [],
            "mape": [],
            "u": []
        }

        for i in range(n_windows):
            train_limit = last_cut + train_length
            test_limit = last_cut + windows_length
            train = data[last_cut:train_limit]
            test = data[train_limit:test_limit]
            print('-' * 20)
            print(f'training window {(last_cut, test_limit)}')
            model.fit(train, dump='time')
            print("[{0: %H:%M:%S}]".format(datetime.datetime.now()) + f" getting statistics")
            rmse, mape, u = Measures.get_point_statistics(test, model)

            result["rmse"].append(rmse)
            result["mape"].append(mape)
            result["u"].append(u)
            result["window"].append((last_cut, test_limit))

            last_cut = last_cut + windows_length
        result = pd.DataFrame(result)
        return result