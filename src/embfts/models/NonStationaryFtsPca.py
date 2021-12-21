from embedding.EmbeddingPCA import EmbeddingPCA
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


    def run_test_target(self, data,steps_ahead):
        forecast = self.forecast(self.prepare_data(data),steps_ahead)
        return forecast, data

    def create_non_stationary(self,data):
        nsfs = nspart.simplenonstationary_gridpartitioner_builder(data=data, npart=self.npart, transformation=None)
        self.model = nsfts.NonStationaryFTS(partitioner=nsfs, order=self.order_fts_model, memory_window = self.memory_window_error)
        #self.model = nsfts.WeightedNonStationaryFTS(partitioner=nsfs, order=self.order_fts_model)

    def fit(self,data):
        self.model.fit(data)

    def forecast(self,data,steps_ahead):
        forecast = self.model.predict(data,steps_ahead=steps_ahead)
        return forecast

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

    def inverse_transformation(self,data,target_column):
        data = np.array(data)
        #inverse_pca = self.pca.inverse_transform(data.reshape(len(data), 1))
        inverse_pca = self.pca.pca_sklearn_inverse(data.reshape(len(data), 1))
        final_result = self.pca.standardization_inverse(inverse_pca)
        return final_result[:, target_column]

    def prepare_data(self,data):
        df = pd.DataFrame(data,columns=['component 1'])
        return df['component 1'].values