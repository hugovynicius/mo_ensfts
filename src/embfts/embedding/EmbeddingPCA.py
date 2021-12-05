import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, KernelCenterer, MinMaxScaler
from sklearn.utils import extmath
from sklearn.metrics.pairwise import euclidean_distances


class EmbeddingPCA():
    def __init__(self, no_of_components,gamma):
        self.no_of_components = no_of_components
        self.eigen_values = None
        self.eigen_vectors = None

        # debug attributes
        self.name = 'Principle Component Analysis'
        self.shortname = 'PCA'

        self.std = StandardScaler()
        self.pca_sk = PCA(n_components=self.no_of_components)
        self.gamma = gamma
        self.kpca_sk = KernelPCA(n_components=self.no_of_components, kernel='rbf', gamma=gamma, fit_inverse_transform = True)

    def __str__(self):
        return self.name

    def transform(self, x):
        return np.dot(x - self.mean, self.projection_matrix.T)

    def inverse_transform(self, x):
        return np.dot(x, self.projection_matrix) + self.mean

    def fit(self, x):
        self.no_of_components = x.shape[1] if self.no_of_components is None else self.no_of_components
        self.mean = np.mean(x, axis=0)

        cov_matrix = np.cov(x - self.mean, rowvar=False)

        self.eigen_values, self.eigen_vectors = np.linalg.eig(cov_matrix)
        self.eigen_vectors = self.eigen_vectors.T

        self.sorted_components = np.argsort(self.eigen_values)[::-1]

        self.projection_matrix = self.eigen_vectors[self.sorted_components[:self.no_of_components]]
        self.explained_variance = self.eigen_values[self.sorted_components];
        self.explained_variance_ratio = self.explained_variance / self.eigen_values.sum()

    def pca_sklearn(self,x):
        return self.pca_sk.fit_transform(x);

    def pca_sklearn_inverse(self, x):
        return self.pca_sk.inverse_transform(x);

    def standardization(self, x):
        return self.std.fit_transform(x)

    def standardization_inverse(self,x):
        return self.std.inverse_transform(x)

    def kernel_pca(self,x, gamma):
        # Calculate euclidean distances of each pair of points in the data set
        dist = euclidean_distances(x, x, squared=True)

        # Calculate Kernel matrix
        k = np.exp(-gamma * dist)
        kc = KernelCenterer().fit_transform(k)

        # Get eigenvalues and eigenvectors of the kernel matrix
        eig_vals, eig_vecs = np.linalg.eigh(kc)

        # flip eigenvectors' sign to enforce deterministic output
        eig_vecs, _ = extmath.svd_flip(eig_vecs, np.empty_like(eig_vecs).T)

        # Concatenate the eigenvectors corresponding to the highest n_components eigenvalues
        xkpca = np.column_stack([eig_vecs[:, -i] for i in range(1, self.no_of_components + 1)])

        return xkpca

    def kernel_pca_sklearn(self, x, gamma):
        #self.kpca_sk = KernelPCA(n_components=self.no_of_components, kernel='rbf', gamma=gamma)
        return self.kpca_sk.fit_transform(x)

    def kernel_pca_sklearn_inverse(self, x, gamma):
        return self.kpca_sk.inverse_transform(x)
