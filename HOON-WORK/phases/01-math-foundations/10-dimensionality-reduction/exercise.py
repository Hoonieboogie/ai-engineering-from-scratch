"""
Q1) Modify the PCA class to support inverse_transform.
Reconstruct MNIST digits from 10, 50, and 200 components.
Print the reconstruction error (mean squared difference from the original) for each.
"""
print("<Q1: PCA Intro>")
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
from sklearn.datasets import fetch_openml, make_classification
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
n_components_list = [10, 50, 200]
for n_components in n_components_list:
    pca_mnst = PCA(n_components)
    transformed_X = pca_mnst.fit_transform(mnist['data']) # transform
    reconstructed_X = pca_mnst.inverse_transform(transformed_X)

    mse = np.mean((mnist["data"] - reconstructed_X) ** 2)
    print(mse)


"""
Q2) Run t-SNE on the same MNIST subset with perplexity values of 5, 30, and 100.
Describe how the output changes.
Why does perplexity affect cluster tightness?
"""
print("="*100)
print("<Q2: t-SNE>")
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

perplexities = [5, 30, 100]

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_mnist = mnist.data[:5000].astype(float)
y_mnist = mnist.target[:5000].astype(int)
results = {}

for perplexity in perplexities:
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    results[perplexity] = tsne.fit_transform(X_mnist)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, perplexity in zip(axes, perplexities):
    X_tsne = results[perplexity]

    ax.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=y_mnist,
        s=5,
        cmap="tab10",
    )
    ax.set_title(f"Perplexity = {perplexity}")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

# Why perplexity affects cluster tightness?
## Low perplexity: focuses on a few closest neighbors, often creating tight or fragmented clusters.
## High perplexity: considers more neighbors, producing broader, smoother, or more connected clusters.


"""
Q3) Take a dataset with 50 features
where only 5 are informative (generate one with sklearn.datasets.make_classification).
Apply PCA and check whether the explained variance curve correctly identifies
that the data is effectively 5-dimensional.
"""
print("="*100)
print("<Q3: PCA Application>")
X_q3, y_q3 = make_classification(
    n_samples=1000,
    n_features=50,
    n_informative=5,
    n_redundant=0,
    n_repeated=0,
    random_state=42,
)
print(f"X shape: {X_q3.shape}")
print(f"y shape: {y_q3.shape}")

pca_q3 = PCA(n_components=50)
pca_q3.fit(X_q3)
# Cumulative varinace: how much of the dataset’s total variance is preserved if we keep the first k PCA components?
cumulative_variance = np.cumsum(pca_q3.explained_variance_ratio_)

for component, variance in enumerate(cumulative_variance, start=1):
    print(f"components={component:>2}  cumulative_variance={variance:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(range(1, 51), cumulative_variance, marker=".")
plt.axvline(5, color="red", linestyle="--", label="5 informative features")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative explained variance")
plt.title("Q3: Explained variance for a 50-feature dataset")
plt.legend()
plt.tight_layout()
plt.show()

# Q3 Conclusion
# The curve rises gradually and has no sharp elbow near k=5.
# The first five components preserve about 20% of the variance.
# The variance is spread across many directions, not concentrated in only five.
# Therefore, PCA does not discover a clear five-dimensional structure here.
# The five features are informative for predicting y, but PCA ignores y and only measures variance in X.
