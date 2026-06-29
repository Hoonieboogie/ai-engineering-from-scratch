## Vectors and Matrices Operations Exercises


# ============================ Utils ============================
from operator import inv

from sympy import det


class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x**2 for x in self.data) ** 0.5


class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix(
            [
                [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ]
        )

    def __sub__(self, other):
        return Matrix(
            [
                [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ]
        )

    def scalar_multiply(self, scalar):
        return Matrix(
            [
                [self.data[i][j] * scalar for j in range(self.cols)]
                for i in range(self.rows)
            ]
        )

    def element_wise_multiply(self, other):
        return Matrix(
            [
                [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ]
        )

    def matmul(self, other):
        return Matrix(
            [
                [
                    sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                    for j in range(other.cols)
                ]
                for i in range(self.rows)
            ]
        )

    def transpose(self):
        return Matrix(
            [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)]
        )

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix(
                [
                    [self.data[i][k] for k in range(self.cols) if k != j]
                    for i in range(1, self.rows)
                ]
            )
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix(
            [
                [self.data[1][1] / det, -self.data[0][1] / det],
                [-self.data[1][0] / det, self.data[0][0] / det],
            ]
        )

    def inverse_3x3(self):
        """Inverse via the adjugate method: A^-1 = (1/det) * transpose(cofactors)."""
        # Step 1) guard: a zero-determinant matrix has no inverse
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")

        # Step 2) cofactor[i][j] = (-1)^(i+j) * det(minor at i,j)
        #         minor = this matrix with row i and column j removed
        cofactors = []
        for i in range(3):
            row = []
            for j in range(3):
                minor = Matrix(
                    [
                        [
                            self.data[r][c] for c in range(3) if c != j
                        ]  # keep all columns except j
                        for r in range(3)
                        if r != i  # keep all rows except i
                    ]
                )
                row.append(((-1) ** (i + j)) * minor.determinant())
            cofactors.append(row)

        # Step 3) adjugate = transpose of the cofactor matrix
        adjugate = Matrix(cofactors).transpose()

        # Step 4) inverse = adjugate / det
        return adjugate.scalar_multiply(1 / det)

    @staticmethod
    def identity(n):
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])


# ================================================================

"""
Q1) Verify the inverse. Multiply A @ A.inverse_2x2() and confirm you get the identity matrix.
Try it with three different 2x2 matrices. What happens when the determinant is zero?
"""

test_matrices = [
    Matrix([[2, 1], [1, 1]]),
    Matrix([[1, 1], [1, 1]]),
    Matrix([[5, 3], [10, 3]]),
]

for matrix in test_matrices:
    try:
        print(matrix.matmul(matrix.inverse_2x2()))
    except ValueError as e:
        print("No inverse (det=0):", e)
print("-" * 100)


"""
Q2) Implement 3x3 inverse. Extend the Matrix class to compute inverses for 3x3
matrices using the adjugate method. Test it against NumPy's np.linalg.inv.
"""
# Check what I did to Matrix class (method implemented)
q2_test_mat = Matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
print(q2_test_mat.inverse_3x3())
print("-" * 100)

"""
Q3) Build a two-layer network. Using only your Matrix class (no NumPy),
create a two-layer neural network: input (3) -> hidden (4) -> output (2).
Initialize random weights, run a forward pass, and verify all shapes are correct.
"""
# input shape: 1*3
# output shape: 1*2

import random

input = Matrix([[1, 2, 3]])  # 1*3
W1 = Matrix([[random.random() for _ in range(4)] for _ in range(3)])  # 3*4
W2 = Matrix([[random.random() for _ in range(2)] for _ in range(4)])  # 4*2

hidden_layer = input.matmul(W1)
output = hidden_layer.matmul(W2)

print("input :", input.shape)  # (1, 3)
print("W1    :", W1.shape)  # (3, 4)
print("hidden:", hidden_layer.shape)  # (1, 4)
print("W2    :", W2.shape)  # (4, 2)
print("output:", output.shape)  # (1, 2)
