from pyFTS.data import artificial
import numpy as np

class GenerateConceptDriftDatasets():
    def __init__(self, deflen, order):
        self.deflen = deflen
        self.totlen = self.deflen * 10
        self.order = order

        # debug attributes
        self.name = 'Generate Concept Drift Datasets'
        self.shortname = 'Drift Datasets'

    def mavg(self,l, order=2):
        ret = []  # l[:order]
        for k in np.arange(order, len(l)):
            ret.append(np.nanmean(l[k - order:k]))
        return ret

    def generate_stationary_signal(self, qtd_drifts, mu_local, sigma_local):
        drift = []
        for i in range(qtd_drifts):
            signal = artificial.SignalEmulator().stationary_gaussian(mu_local, sigma_local, length=self.deflen,
                                                                     it=10).run()
            drift = np.concatenate((drift, signal))
        return self.mavg(drift, self.order)

    def generate_stationary_signal_with_blip(self, qtd_drifts, mu_local, sigma_local):
        drift = []
        for i in range(qtd_drifts):
            signal = artificial.SignalEmulator().stationary_gaussian(mu_local, sigma_local, length=self.deflen,
                                                                     it=10).blip().blip().run()
            drift = np.concatenate((drift, signal))
        return self.mavg(drift, self.order)

    def generate_sudden_variance(self, qtd_drifts, sigma_drift, mu_local, sigma_local):
        drift = []
        for i in range(qtd_drifts):
            signal = artificial.SignalEmulator() \
                .stationary_gaussian(mu_local, sigma_local, length=self.deflen // 2, it=10) \
                .stationary_gaussian(mu_local, sigma_drift, length=self.deflen // 2, it=10, additive=False) \
                .run()
            drift = np.concatenate((drift, signal))
        return self.mavg(drift, self.order)

    def generate_sudden_mean(self, qtd_drifts, mu_drift, mu_local, sigma_local):
        drift = []
        for i in range(qtd_drifts):
            signal = artificial.SignalEmulator() \
                .stationary_gaussian(mu_local, sigma_local, length=self.deflen // 2, it=10) \
                .stationary_gaussian(mu_drift, sigma_local, length=self.deflen // 2, it=10, additive=False) \
                .run()
            drift = np.concatenate((drift, signal))
        return self.mavg(drift, self.order)

    def generate_sudden_mean_variance(self, qtd_drifts, mu_drift, sigma_drift, mu_local, sigma_local):
        drift = []
        for i in range(qtd_drifts):
            signal = artificial.SignalEmulator() \
                .stationary_gaussian(mu_local, sigma_local, length=self.deflen // 2, it=10) \
                .stationary_gaussian(mu_drift, sigma_drift, length=self.deflen // 2, it=10, additive=False) \
                .run()
            drift = np.concatenate((drift, signal))
        return self.mavg(drift, self.order)

    def generate_incremental_mean(self, qtd_drifts, mu_local, sigma_local):
        drift = []
        for i in range(qtd_drifts):
            signal = artificial.SignalEmulator() \
                .stationary_gaussian(mu_local, sigma_local, length=self.deflen, it=10) \
                .incremental_gaussian(0.1, 0, length=self.totlen // 2, start=self.totlen // 2) \
                .run()
            drift = np.concatenate((drift, signal))
        return self.mavg(drift, self.order)

    def generate_incremental_variance(self, qtd_drifts, mu_drift, sigma_drift, mu_local, sigma_local):
        drift = []
        for i in range(qtd_drifts):
            signal = artificial.SignalEmulator() \
                .stationary_gaussian(mu_local, sigma_local, length=self.deflen, it=10) \
                .incremental_gaussian(0., 0.1, length=self.totlen // 2, start=self.totlen // 2) \
                .run()
            drift = np.concatenate((drift, signal))
        return self.mavg(drift, self.order)

    def generate_incremental_mean_variance(self, qtd_drifts, mu_local, sigma_local):
        drift = []
        for i in range(qtd_drifts):
            signal = artificial.SignalEmulator() \
                .stationary_gaussian(mu_local, sigma_local, length=self.deflen, it=10) \
                .incremental_gaussian(0.02, 0.01, length=self.totlen // 2, start=self.totlen // 2) \
                .run()
            drift = np.concatenate((drift, signal))
        return self.mavg(drift, self.order)

