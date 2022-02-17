import numpy as np
import pandas as pd
from pyFTS.models.nonstationary import partitioners as nspart
from pyFTS.models.nonstationary import nsfts
from embfts.embedding.EmbeddingPCA import EmbeddingPCA

class MimoNonStationaryFtsPca():
    def __init__(self, num_components_pca, order_fts_model, npart,
                        gamma, fts_model, memory_window_error):
        # debug attributes
        self.name = 'Mimo Non Stationary Emebeding Fuzzy Time Series '
        self.shortname = 'MIMO ENSFTS'

        ## Parameters of the model
        self.num_components_pca = num_components_pca
        self.fts_model = fts_model
        self.order_fts_model = order_fts_model
        self.model = self.fts_model
        self.memory_window_error = memory_window_error
        self.gamma = gamma
        self.npart = npart

        # PCA
        self.pca = EmbeddingPCA(no_of_components=self.num_components_pca, gamma = self.gamma)

    def run_test(self, models, data,steps_ahead,transformation):
        if transformation == 'PCA':
            pca_test = self.create_pca_components(data)
        else:
            pca_test = self.create_kernel_pca_components_test(data)

        components = pca_test.columns
        forecast_model = []

        for i in range(0, self.num_components_pca):
            mf = models[i].forecast(pca_test.loc[:,components[i]], steps_ahead=steps_ahead)
            if (i == 0):
                forecast_model = mf
            else:
                forecast_model = np.column_stack((forecast_model,mf))

        if transformation == 'PCA':
            forecast = self.inverse_transformation(forecast_model)
        else:
            forecast = self.inverse_kpca_transformation(forecast_model)

        return forecast, data

    def run_train(self, data,transformation):
        if transformation == 'PCA':
            pca_train = self.create_pca_components(data)
        else:
            pca_train = self.create_kernel_pca_components(data)
        models = self.create_and_fit_mimo_models(pca_train)
        return models,pca_train

    def create_and_fit_mimo_models(self, data):
        components = data.columns
        models = []

        for i in range(0, self.num_components_pca):
            md = self.create_non_stationary(data.loc[:,components[i]])
            md.fit(data.loc[:,components[i]])
            models.append(md)

        return models

    def create_non_stationary(self,data):
        nsfs = nspart.simplenonstationary_gridpartitioner_builder(data=data, npart=self.npart,
                                                                  transformation=None)
        model = nsfts.NonStationaryFTS(partitioner=nsfs, order=self.order_fts_model,
                                            memory_window = self.memory_window_error,window_size = 3)

        return model

    def create_pca_components(self,data):
        transformed = self.pca.standardization(data)
        x_std = self.pca.pca_sklearn(transformed)

        components = []
        for i in range(0, self.num_components_pca):
            components.append('C' + str(i))

        df_pca = pd.DataFrame(x_std, columns=list(components))
        return df_pca

    def create_names_compoments(self):
        components = None
        for i in range (0,self.num_components_pca):
            components.append()

    def create_kernel_pca_components(self,data):
        transformed = self.pca.standardization(data)
        x_std = self.pca.kernel_pca_sklearn(transformed,self.gamma)

        components = []
        for i in range(0, self.num_components_pca):
            components.append('C' + str(i))

        df_kpca = pd.DataFrame(x_std, columns=list(components))
        return df_kpca

    def create_kernel_pca_components_test(self,data):
        transformed = self.pca.standardization_test(data)
        x_std = self.pca.kernel_pca_sklearn(transformed,self.gamma)

        components = []
        for i in range(0, self.num_components_pca):
            components.append('C' + str(i))

        df_kpca = pd.DataFrame(x_std, columns=list(components))
        return df_kpca

    def inverse_transformation(self, data):
        inverse_pca = self.pca.pca_sklearn_inverse(data)
        final_result = self.pca.standardization_inverse(inverse_pca)
        return final_result

    def inverse_kpca_transformation(self, data):
        inverse_kpca = self.pca.kernel_pca_sklearn_inverse(data,self.gamma)
        final_result = self.pca.standardization_inverse(inverse_kpca)
        return final_result

    def prepare_data(self,data):
        df = pd.DataFrame(data,columns=['component'])
        return df['component'].values