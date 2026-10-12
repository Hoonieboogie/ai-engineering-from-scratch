"""
Q1) Implement the full SVD from scratch without using power iteration.
Instead, compute the eigendecomposition of A^T A to get V and the singular values,
then compute U = A V Sigma^{-1}.
Compare numerical accuracy with your power iteration version and with NumPy.
"""

import numpy as np
from svd import reconstruct, svd_from_scratch as svd_via_power_iteration


# Math-to-code map:
# A^T A = V diag(sigma_i^2) V^T, so eigendecomposition gives V and sigma_i^2.
# Sort eigenvalue/eigenvector pairs so sigma_1 >= sigma_2 >= ... .
# sigma_i = sqrt(eigenvalue_i).
# A v_i = sigma_i u_i, therefore u_i = (A v_i) / sigma_i.
# Finally, A is reconstructed as U diag(S) V^T.
def svd_via_eigendecomposition(A, k=None):
    m, n = A.shape
    if k is None:
        k = min(m, n)

    eigenvalues, eigenvectors = np.linalg.eigh(
        A.T @ A
    )  # return eigenvalues in asc order
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    singular_values = np.sqrt(eigenvalues)

    selected_values = singular_values[:k]
    nonzero = selected_values > 1e-10

    S = selected_values[nonzero]
    V = eigenvectors[:, :k][:, nonzero]
    U = (A @ V) / S

    return U, S, V


def reconstruction(U, S, V):
    return U @ np.diag(S) @ V.T


"""
Q2) Load a real grayscale image (or convert one to grayscale).
Compress it at ranks 1, 5, 10, 25, 50, 100.
For each rank, compute the compression ratio and the relative error.
Find the rank where the image becomes visually acceptable.
"""


def compression_ratio(m, n, k):
    original = m * n
    compressed = k * (m + n + 1)
    return compressed / original


def truncated_svd(A, k):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k], S[:k], Vt[:k, :]


if __name__ == "__main__":
    # Q1
    print("<Q1: SVD with Eigendecompostion>")
    np.random.seed(42)
    A = np.random.randn(6, 4)

    U_eig, S_eig, V_eig = svd_via_eigendecomposition(A)
    U_power, S_power, V_power = svd_via_power_iteration(A)
    U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

    A_eig = reconstruction(U_eig, S_eig, V_eig)
    A_power = reconstruction(U_power, S_power, V_power)
    A_np = U_np @ np.diag(S_np) @ Vt_np

    print("Singular values:")
    print("  eigendecomposition:", np.round(S_eig, 6))
    print("  power iteration:   ", np.round(S_power, 6))
    print("  NumPy:              ", np.round(S_np, 6))

    print("\nReconstruction errors:")
    print("  eigendecomposition:", np.linalg.norm(A - A_eig))
    print("  power iteration:   ", np.linalg.norm(A - A_power))
    print("  NumPy:              ", np.linalg.norm(A - A_np))

    # Q2
    print("=" * 100)
    print("<Q2: Image Compression with SVD")
    compress_ranks = [1, 5, 10, 25, 50, 100]

    from sklearn.datasets import load_sample_image
    import matplotlib.pyplot as plt

    image = load_sample_image("china.jpg")
    grayscale = image.astype(float).mean(axis=2)
    m, n = grayscale.shape
    # image view
    # plt.imshow(grayscale, cmap="gray", vmin=0, vmax=255)
    # plt.axis("off")
    # plt.show()
    print(m, n)  # 427 X 640

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for ax, compress_rank in zip(axes.flat, compress_ranks):
        comp_ratio = compression_ratio(m, n, compress_rank)
        U, S, Vt = truncated_svd(grayscale, compress_rank)

        # Reconstruct
        reconstructed_image = reconstruction(U, S, Vt.T)
        ax.imshow(reconstructed_image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Rank = {compress_rank}")
        ax.axis("off")

        relative_error = np.linalg.norm(
            grayscale - reconstructed_image, "fro"
        ) / np.linalg.norm(grayscale, "fro")
        print(
            f"Rank {compress_rank:3d}: storage ratio = {comp_ratio:.2%}, "
            f"relative error = {relative_error:.2%}"
        )

    fig.tight_layout()
    plt.show()

    # Q2 conclusion: Rank 25 is visually acceptable for this image.
    # It retains the main scene while using about 9.77% of the original value count.
    # Its relative reconstruction error is about 12.84%.
    # Higher ranks preserve more detail but require more stored values.
    # The storage ratio compares value counts, not encoded image file sizes.
    print("Visually acceptable rank: 25")
