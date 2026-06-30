## Matrix Transformations Exercises

"""
Q1) Apply rotation, scaling, and shearing to a unit square
    (corners at [0,0], [1,0], [1,1], [0,1]).
    Print the transformed corners for each.
    Verify that rotation preserves distances between corners.
"""

import numpy as np

unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # 4 * 2
# Rotate 90 degrees
theta = np.pi / 2
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  # 2 * 2
rotated_square = (R @ unit_square.T).T  # .T at the end for readability


# Distance verification
def dist(p1, p2):
    return np.sqrt(sum((p1 - p2) ** 2))


print(dist(unit_square[0], unit_square[1]))
print(dist(rotated_square[0], rotated_square[1]))

# Scaling
S = np.array([[2, 0], [0, 0.5]])
scaled_square = (S @ unit_square.T).T

# Shearing: Horizonsal shear with k=1
Sh = np.array([[1, 1], [0, 1]])
sheared_square = (Sh @ unit_square.T).T

# Final output
print("rotated: ", rotated_square)
print("scaled:  ", scaled_square)
print("sheared: ", sheared_square)
print("=" * 100)


"""
Q2) Find the eigenvalues of the matrix [[4, 2], [1, 3]] by hand using the
    characteristic equation. Then verify with your from-scratch function
    and with NumPy.
"""
A = np.array([[4, 2], [1, 3]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(A)

import math


def my_eigenvalue_2X2(A):
    a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
    # characteristic equation: λ² - (a+d)λ + (ad-bc) = 0
    trace = a + d
    det = a * d - b * c
    discriminant = trace**2 - 4 * det
    return (trace + math.sqrt(discriminant)) / 2, (trace - math.sqrt(discriminant)) / 2


print(f"\nEigenvalues: {eigenvalues}, {my_eigenvalue_2X2([[4, 2], [1, 3]])}")
print(f"Eigenvectors (columns):\n{eigenvectors}")
print("=" * 100)

"""
Q3) Create a composition of three transformations
    (rotate 30 degrees, scale by [1.5, 0.8], shear with kx=0.3)
    and apply it to 8 points arranged in a circle.
    Print before and after coordinates.
    Compute the determinant of the composed matrix and verify it equals
    the product of the individual determinants.
"""
q3_theta = np.pi / 6
q3_R = np.array(
    [[np.cos(q3_theta), -np.sin(q3_theta)], [np.sin(q3_theta), np.cos(q3_theta)]]
)  # 2X2
q3_S = np.array([[1.5, 0], [0, 0.8]])  # 2X2
q3_SH = np.array([[1, 0.3], [0, 1]])  # 2X2

C = q3_SH @ q3_S @ q3_R

# generate 8 points on a circle
angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)  # 1 * 8
circle_points = np.array([[np.cos(t), np.sin(t)] for t in angles])  # shape (8, 2)
composed_applied = (C @ circle_points.T).T

# Verification of composition
if np.linalg.det(C) == np.linalg.det(q3_R) * np.linalg.det(q3_S) * np.linalg.det(q3_SH):
    print("Correct")
else:
    print("Wrong")


print("before:", circle_points)
print("after: ", (C @ circle_points.T).T)
