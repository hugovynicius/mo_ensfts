import pandas as pd
import numpy as np
from pyFTS.benchmarks import Measures
import math
from pyFTS.common import Util
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import statistics
from embfts.util.MetricsUtil import MetricsUtil
from embfts.util.PlotUtil import PlotUtil


class StatisticsUtil():
    def __init__(self):
        # debug attributes
        self.name = 'Statistics Util'
        self.shortname = 'stutil'

        self.metricsUtil = MetricsUtil()
        self.plot = PlotUtil()

    def sliding_windows_miso_ensfts(self, data, n_windows, train_size, model, transformation, first_col_train, last_col_train, target_col_train,
                                    target_col_test, plot_graph = False, steps_ahead=1):

        result = {
            "rmse": [],
            "mape": [],
            "smape": [],
            "mae": [],
            "r2": [],
            "nrmse": []
        }

        final_result = {
            "rmse": [],
            "mape": [],
            "smape": [],
            "mae": [],
            "r2": [],
            "nrmse": []
        }

        tam = len(data)
        windows_length = math.floor(tam / n_windows)

        for ct, ttrain, ttest in Util.sliding_window(data, windows_length, train_size, inc=1):
            if len(ttest) > 0:
                print('-' * 20)
                print(f'training window {(ct)}')
                ttrain = ttrain.loc[:, first_col_train:last_col_train]
                model_train, pca_train = model.run_train(ttrain,transformation)
                y_test = ttest[target_col_train].values
                forecast, data_test = model.run_test_target(y_test, steps_ahead)
                # ttest_test = ttest.loc[:, first_col_train:last_col_train]
                # forecast = model.run_test(ttest_test,transformation,target_col_train,steps_ahead)

                y_validation = ttest[target_col_test].values

                y_validation = y_validation[:len(y_validation) - 1]
                forecast = forecast[1:]

                result = self.metricsUtil.compute_all_metrics(y_validation, forecast, result)

                if plot_graph == True:
                    self.plot.plot_orginal_forecast(y_validation,forecast)

        measures = pd.DataFrame(result)
        final_result["rmse"].append(statistics.mean(measures['rmse']))
        final_result["mape"].append(statistics.mean(measures['mape']))
        final_result["smape"].append(statistics.mean(measures['smape']))
        final_result["mae"].append(statistics.mean(measures['mae']))
        final_result["r2"].append(statistics.mean(measures['r2']))
        final_result["nrmse"].append(statistics.mean(measures['nrmse']))

        final_measures = pd.DataFrame(final_result)

        return final_measures, measures

    def sliding_window_mimo_ensfts(self, data, n_windows, train_size, model, transformation, first_col_train,
                                   last_col_train, first_col_test, last_col_test, df_forecats_columns, plot_graph = False, steps_ahead=1 ):

        result = {
            "window": [],
            "rmse": [],
            "mape": [],
            "mae": [],
            "r2": [],
            "nrmse": [],
            "variable": []
        }

        final_result = {
            "window": [],
            "rmse": [],
            "mape": [],
            "mae": [],
            "r2": [],
            "nrmse": [],
            "variable": []
        }

        tam = len(data)
        n_windows = n_windows
        windows_length = math.floor(tam / n_windows)
        for ct, ttrain, ttest in Util.sliding_window(data, windows_length, train_size, inc=1):
            if len(ttest) > 0:

                print('-' * 20)
                print(f'training window {(ct)}')

                df_train = ttrain.loc[:, first_col_train:last_col_train]
                df_test = ttest.loc[:, first_col_train:last_col_train]
                df_original = ttest.loc[:, first_col_test:last_col_test]

                models, data_train = model.run_train(df_train, transformation)
                forecast, data_test = model.run_test(models, df_test, steps_ahead, transformation)

                columns = list(df_forecats_columns)
                df_forecast = pd.DataFrame(forecast, columns=columns)

                for col in columns:
                    original = df_original[col].values
                    forecast = df_forecast[col].values
                    original = original[:len(original) - 1]
                    forecast = forecast[1:]

                    mae = mean_absolute_error(original, forecast)
                    r2 = r2_score(original, forecast)
                    rmse = Measures.rmse(original, forecast)
                    mape = Measures.mape(original, forecast)
                    nrmse = Measures.nrmse(original, forecast)

                    result["rmse"].append(rmse)
                    result["nrmse"].append(nrmse)
                    result["mape"].append(mape)
                    result["mae"].append(mae)
                    result["r2"].append(r2)
                    result["nrmse"].append(nrmse)
                    result["window"].append(ct)
                    result["variable"].append(col)

                    if(plot_graph == True):
                        self.plot.plot_orginal_forecast(original,forecast)

        measures = pd.DataFrame(result)

        columns = list(df_forecats_columns)

        final_result_mimo = {
            "variable": [],
            "rmse": [],
            "mae": [],
            "mape": [],
            "r2": [],
            "nrmse": []
        }

        var = measures.groupby("variable")

        for col in columns:
            var_agr = var.get_group(col)

            rmse = round(statistics.mean(var_agr.loc[:, 'rmse']), 3)
            mape = round(statistics.mean(var_agr.loc[:, 'mape']), 3)
            mae = round(statistics.mean(var_agr.loc[:, 'mae']), 3)
            r2 = round(statistics.mean(var_agr.loc[:, 'r2']), 3)
            nrmse = round(statistics.mean(var_agr.loc[:, 'nrmse']), 3)

            final_result_mimo["variable"].append(col)
            final_result_mimo["rmse"].append(rmse)
            final_result_mimo["mape"].append(mape)
            final_result_mimo["mae"].append(mae)
            final_result_mimo["r2"].append(r2)
            final_result_mimo["nrmse"].append(nrmse)

        final_measures = pd.DataFrame(final_result)

        return final_measures
