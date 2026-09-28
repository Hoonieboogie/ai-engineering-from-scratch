"""
Q1) Implement the full SVD from scratch without using power iteration.
Instead, compute the eigendecomposition of A^T A to get V and the singular values,
then compute U = A V Sigma^{-1}.
Compare numerical accuracy with your power iteration version and with NumPy.
"""

import numpy as np
from svd import svd_from_scratch as svd_via_power_iteration


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
