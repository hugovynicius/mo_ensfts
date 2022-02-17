from embfts.embedding.EmbeddingPCA import EmbeddingPCA
import numpy as np
import pandas as pd
from pyFTS.models.nonstationary import partitioners as nspart
from pyFTS.models.nonstationary import nsfts

class NonStationaryFtsPca():
    def __init__(self, num_components_pca, fts_model, order_fts_model, npart, gamma,memory_window_error):
        # debug attributes
        self.name = 'Non Stationary FTS-PCA'
        self.shortname = 'Non Stationary FTS-PCA'

        ## Parameters of the model
        self.num_components_pca = num_components_pca
        self.fts_model = fts_model
        self.order_fts_model = order_fts_model
        self.model = self.fts_model
        self.memory_window_error = memory_window_error
        self.gamma = gamma

        # PCA
        self.pca = EmbeddingPCA(no_of_components=self.num_components_pca, gamma = self.gamma)

        # Hyperparameter kernel PCA
        self.gamma = gamma

        # Number of fuzzy sets
        self.npart = npart

    def run_train_kernel_pca_non_stationary(self,data):
        #pca_train = self.create_pca_components(data)
        pca_train = self.create_kernel_pca_components(data)
        self.create_non_stationary(pca_train)
        self.fit(self.prepare_data(pca_train))
        return self.model, pca_train

    def run_train_pca_non_stationary(self, data):
        pca_train = self.create_pca_components(data)
        self.create_non_stationary(pca_train)
        self.fit(self.prepare_data(pca_train))
        return self.model, pca_train

    def run_train(self,data,transformation):
        pca_train = self.create_components(data,transformation)
        self.create_non_stationary(pca_train)
        self.fit(self.prepare_data(pca_train))
        return self.model, pca_train

    def run_test(self,data,transformation,target_column, steps_ahead):
        if transformation == 'PCA':
            pca_test = self.create_pca_components(data)
        else:
            pca_test = self.create_kernel_pca_components(data)
        forecast = self.forecast(self.prepare_data(pca_test), steps_ahead)
        forecast = self.inverse_transformation(forecast,transformation,data.columns,target_column)
        return forecast

    def run_test_target(self, data,steps_ahead):
        #data = self.pca.standardization(data.reshape(-1, 1))
        forecast = self.forecast(self.prepare_data(data),steps_ahead)
        return forecast, data

    def create_non_stationary(self, data):
        from pyFTS.common import Transformations
        tdiff = Transformations.Differential(1)
        boxcox = Transformations.BoxCox(1)

        nsfs = nspart.simplenonstationary_gridpartitioner_builder(data=data, npart=self.npart, transformation=None)
        # self.model = nsfts.NonStationaryFTS(partitioner=nsfs, order=self.order_fts_model,
        #                                     memory_window=self.memory_window_error, window_size=0, no_update=True,
        #                                     method='conditional',time_displacement = 3)

        self.model = nsfts.NonStationaryFTS(partitioner=nsfs, order=self.order_fts_model,
                                            memory_window=self.memory_window_error)
        # self.model = nsfts.WeightedNonStationaryFTS(partitioner=nsfs, order=self.order_fts_model,memory_window = self.memory_window_error)

    def fit(self,data):
        self.model.fit(data)

    def forecast(self,data,steps_ahead):
        #forecast = self.model.predict(data,steps_ahead=steps_ahead)
        forecast = self.model.forecast(data, steps_ahead=steps_ahead)
        return forecast

    def create_components(self,data,transformation):
        transformed = data #self.pca.standardization(data)
        if transformation == 'PCA':
            x_std = self.pca.pca_sklearn(transformed)
            # self.pca.fit(transformed)
            # x_std = self.pca.transform(transformed)
        else:
            x_std = self.pca.kernel_pca_sklearn(transformed, self.gamma)
        return x_std

    def create_pca_components(self,data):
        transformed = self.pca.standardization(data)
        #self.pca.fit(transformed)
        #x_std = self.pca.transform(transformed)
        x_std = self.pca.pca_sklearn(transformed)
        #return x_std.real
        return x_std

    def create_kernel_pca_components(self,data):
        transformed = self.pca.standardization(data)
        x_std = self.pca.kernel_pca_sklearn(transformed,self.gamma)
        return x_std

    def inverse_transformation(self,data,transformation,columns,target_column):
        data = np.array(data)
        if transformation == 'PCA':
            inverse_pca = self.pca.pca_sklearn_inverse(data.reshape(len(data), 1))
        else:
            inverse_pca = self.pca.kernel_pca_sklearn_inverse(data.reshape(len(data), 1),self.gamma)
        final_result = self.pca.standardization_inverse(inverse_pca)
        df_result = pd.DataFrame(final_result,columns=list(columns))
        return df_result[target_column].values

    def prepare_data(self,data):
        df = pd.DataFrame(data,columns=['component 1'])
        return df['component 1'].values
