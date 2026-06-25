## Linear Algebra Intuition Exercises

"""
Q1) Implement Vector.angle_between(other) that returns the angle in degrees between two vectors

* angle = arccos( (a · b) / (|a| · |b|) )   then convert radians → degrees
"""

import math


class Vector:
    def __init__(self, components):
        self.components = components
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def project_onto(self, other):
        scalar = self.dot(other) / other.dot(other)
        return Vector([scalar * x for x in other.components])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return math.sqrt(sum(x**2 for x in self.components))

    def angle_between(self, other):
        return math.degrees(
            math.acos(self.dot(other) / (self.magnitude() * other.magnitude()))
        )

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


vec_a = Vector([1, 2])
vec_b = Vector([2, 1])
print(f"Q1 Anwer: Angle betwee vector a & vecotr B = {vec_a.angle_between(vec_b)}")
print("-" * 100)


"""
Q2) Create a 2D scaling matrix that doubles the x-coordinate and triples the y-coordinate,
then apply it to the vector [1, 1]
"""


class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector(
                [
                    sum(
                        self.rows[i][j] * other.components[j]
                        for j in range(self.shape[1])
                    )
                    for i in range(self.shape[0])
                ]
            )
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(
                    sum(
                        self.rows[i][k] * other.rows[k][j] for k in range(self.shape[1])
                    )
                )
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix(
            [
                [self.rows[j][i] for j in range(self.shape[0])]
                for i in range(self.shape[1])
            ]
        )

    def rank(self):
        # Gaussian elimination: count the non-zero pivot rows.
        rows = [row[:] for row in self.rows]
        m, n = self.shape
        r = 0
        for col in range(n):
            pivot = None
            for row in range(r, m):
                if abs(rows[row][col]) > 1e-10:
                    pivot = row
                    break
            if pivot is None:
                continue
            rows[r], rows[pivot] = rows[pivot], rows[r]
            scale = rows[r][col]
            rows[r] = [x / scale for x in rows[r]]
            for row in range(m):
                if row != r and abs(rows[row][col]) > 1e-10:
                    factor = rows[row][col]
                    rows[row] = [rows[row][j] - factor * rows[r][j] for j in range(n)]
            r += 1
        return r

    def __repr__(self):
        return f"Matrix({self.rows})"


doubling_matrix = Matrix([[2, 0], [0, 3]])
target_vector = Vector([1, 1])
q2_result_vector = doubling_matrix @ target_vector  # Vector
print("Q2 Answer:", q2_result_vector.components)  # print
print("-" * 100)

"""
Q3) Given 5 random word-like vectors (dimension 50), find the two most similar using cosine similarity
"""
import random

random.seed(42)
vectors = [Vector([random.random() for _ in range(50)]) for _ in range(5)]

best_sim = -2
best_pair = None

for i in range(len(vectors)):
    for j in range(i + 1, len(vectors)):
        sim = vectors[i].cosine_similarity(vectors[j])
        if sim > best_sim:
            best_sim = sim
            best_pair = (i, j)  # remember WHICH two

print(
    f"Q3 Answer\nMost similar: vectors {best_pair[0]} and {best_pair[1]} (cosine = {best_sim:.4f})"
)
print("-" * 100)


"""
Q4) Verify that the Gram-Schmidt output is truly orthonormal:
check that every pair has dot product 0 and every vector has magnitude 1

ANSWER (in words):
  "orthonormal" = two separate claims:
    1. ORTHOGONAL -> every PAIR of vectors has dot product 0 (they are perpendicular)
    2. NORMAL     -> every SINGLE vector has magnitude 1 (unit length)
  So we verify both: dot product runs on pairs, magnitude runs on each vector.

  Floating-point caveat: we never get EXACTLY 0 or 1, only tiny crumbs like
  1e-16 off. So we test "close enough" with a tolerance (1e-10), never == .
"""


def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            w = w - w.project_onto(u)  # subtract the shadow on each prior direction
        if w.magnitude() < 1e-10:
            continue  # vector was dependent -> skip
        orthonormal.append(w.normalize())
    return orthonormal


TOL = 1e-10
basis = gram_schmidt([Vector([1, 1, 0]), Vector([1, 0, 1]), Vector([0, 1, 1])])

# 1. ORTHOGONAL: every pair's dot product is ~0
orthogonal = True
for i in range(len(basis)):
    for j in range(i + 1, len(basis)):
        if abs(basis[i].dot(basis[j])) > TOL:
            orthogonal = False

# 2. NORMAL: every vector's magnitude is ~1
normal = all(abs(v.magnitude() - 1) < TOL for v in basis)

print("Q4 Answer")
print(f"  orthogonal (all pairs dot=0): {orthogonal}")
print(f"  normal (all magnitudes=1):    {normal}")
print(f"  => orthonormal: {orthogonal and normal}")
print("-" * 100)


"""
Q5) Create a 3x3 matrix with rank 2. Verify using the rank() method.
Then explain what geometric object the columns span.

Hint: rank = number of linearly INDEPENDENT columns. For a 3x3 matrix to have
rank 2 (not 3), one column must be a combination of the other two -- e.g. make
the 3rd column equal to (col1 + col2), or some other dependent combination.
"""

# TODO 1: build a 3x3 matrix where one column depends on the other two.
# Columns are read top-to-bottom. Here col3 = col1 + col2, so col3 adds no new
# direction -> rank 2. All three columns are nonzero, so the dependence is HIDDEN
# (you can't spot it by eye -- that's exactly why we run rank()).
rank2_matrix = Matrix(
    [
        [1, 0, 1],  # col1=(1,0,1) col2=(0,1,1) col3=(1,1,2)=col1+col2
        [0, 1, 1],
        [1, 1, 2],
    ]
)

# TODO 2: verify the rank is actually 2 (this line already works once you fill in the matrix)
print("Q5 Answer")
print(f"  matrix: {rank2_matrix.rows}")
print(f"  rank:   {rank2_matrix.rank()}")  # should print 2

# TODO 3: explain (in a comment or print) what geometric object the columns span.
#   - 3 independent columns (rank 3) span all of 3D space (a volume)
#   - 2 independent columns (rank 2) span a ______ through the origin
#   - 1 independent column  (rank 1) span a ______ through the origin
geometric_object = "plane"  # what does rank 2 span in 3D?
print(f"  the columns span: {geometric_object}")
print("-" * 100)


"""
Q6) Project the vector [1, 2, 3] onto [1, 1, 1].
What does the result represent geometrically?

formula: proj_b(a) = (a . b / b . b) * b

TODO A (by hand FIRST, then let the code confirm):
  a . b   = 6
  b . b   = 3
  scale   = (a . b) / (b . b) = 2
  result  = scale * [1,1,1] = [2,2,2]
"""

a = Vector([1, 2, 3])
b = Vector([1, 1, 1])

# TODO B: predict the result by hand above, THEN run this line to check yourself.
q6_result = a.project_onto(b)

print("Q6 Answer")
print(f"  proj of {a.components} onto {b.components} = {q6_result.components}")

# TODO C: explain what the result represents geometrically.
#   Hint: [1,1,1] points along the "diagonal" direction. The projection is the
#   shadow of a on that diagonal. The scale factor is also the AVERAGE of a's
#   components -- think about why that is.
geometric_meaning = (
    "The projection is a's shadow on the diagonal direction [1,1,1]. "
    "It is the closest point on that diagonal line to a. "
    "Since the scale factor is 2, which is the mean of [1,2,3], "
    "the result [2,2,2] replaces every component with the average. "
    "The leftover a - proj = [-1,0,1] is the part that deviates from the mean."
)
print(f"  geometric meaning: {geometric_meaning}")
print("-" * 100)
