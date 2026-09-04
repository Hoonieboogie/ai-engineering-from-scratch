"""
Q1) Modify the PCA class to support inverse_transform.
Reconstruct MNIST digits from 10, 50, and 200 components.
Print the reconstruction error (mean squared difference from the original) for each.
"""
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.components = eigenvectors[:, : self.n_components].T
        self.eigenvalues = eigenvalues[: self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.eigenvalues / total_var

        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    # X_hat = X_reduced @ components + mean
    # (n x k) @ (k x d) + (d,) = (n x d)
    def inverse_transform(self, X_reduced):
        return X_reduced @ self.components + self.mean

# Reconstruct MNIST digits (10, 50, 200 components)
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
n_components_list = [10, 50, 200]
for n_components in n_components_list:
    pca_mnst = PCA(n_components)
    transformed_X = pca_mnst.fit_transform(mnist['data']) # transform
    reconstructed_X = pca_mnst.inverse_transform(transformed_X)

    mse = np.mean((mnist["data"] - reconstructed_X) ** 2)
    print(mse)
