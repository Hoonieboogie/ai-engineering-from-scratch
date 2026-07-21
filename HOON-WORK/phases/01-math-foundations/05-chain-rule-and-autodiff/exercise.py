"""
Q1) Add __pow__ to the Value class so you can compute x ** n.
Verify that d/dx(x^3) at x=2 equals 12.0.
"""
class Value:
    def __init__(self, data, children=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, n):
        """
        Q1)
        """
        out = Value(self.data**n, (self,), f"**{n}")

        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return (
            self * (other**-1)
            if isinstance(other, Value)
            else self * (Value(other) ** -1)
        )

    def relu(self):
        out = Value(max(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """
        Q2)
        """
        import math

        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        import math

        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        import math

        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


"""
Q3)
Build a computation graph for a single neuron: `y = relu(w1*x1 + w2*x2 + b)`.
Compute all five gradients and verify against PyTorch.
"""
x1 = Value(2.0)
x2 = Value(3.0)
w1 = Value(5.0)
w2 = Value(10.0)
b = Value(7.0)
z = w1*x1 + w2*x2 + b

print("z:", z)
y = z.relu()
print("y:", y)

y.backward()
print(f"b.grad: {b.grad}, w1.grad: {w1.grad}, w2.grad: {w2.grad}, x1.grad: {x1.grad}, x2.grad: {x2.grad}")

# z = w1*x1 + w2*x2 + b
# y = relu(z)
# dy/dz = relu'(z)
# Back Propagation
# dy/dw1 = dy/dz * dz/dw1 = relu'(z) * x1
# dy/dx1 = dy/dz * dz/dx1 = relu'(z) * w1
# dy/dw2 = dy/dz * dz/dw2 = relu'(z) * x2
# dy/dx2 = dy/dz * dz/dx2 = relu'(z) * w2
# dy/db = dy/dz * dz/db = relu'(z) * 1

# Pytorch check
import torch
x1_t = torch.tensor(2.0, requires_grad=True)
x2_t = torch.tensor(3.0, requires_grad=True)
w1_t = torch.tensor(5.0, requires_grad=True)
w2_t = torch.tensor(10.0, requires_grad=True)
b_t = torch.tensor(7.0, requires_grad=True)
z_t = w1_t * x1_t + w2_t * x2_t + b_t
y_t = torch.relu(z_t)
y_t.backward()

print(w1_t.grad)
print(x1_t.grad)
print(w2_t.grad)
print(x2_t.grad)
print(b_t.grad)
print("=========================================================")
"""
Q4) Implement forward-mode autodiff using dual numbers.
Create a `Dual` class and verify it gives the same derivatives as your reverse-mode engine.
"""
class Dual:
    def __init__(self, data, derivative=0.0):
        self.data = float(data)
        self.derivative = float(derivative)

    def __add__(self, other):
        out = Dual(self.data + other.data, self.derivative + other.derivative)
        return out

    def __mul__(self, other):
        # (a + Δa)(b + Δb) = ab + bΔa + aΔb + ΔaΔb
        # Δ(ab) = bΔa + aΔb + ΔaΔb
        # For tiny changes, ignore ΔaΔb: d(ab)/dx = a'*b + a*b'.
        out = Dual(self.data*other.data ,self.derivative * other.data + self.data * other.derivative)
        return out

x = Dual(2,1)
c = Dual(3,0)
l = Dual(5.0)
k1 = x + c
k2 = x * c
print(k1.data, k1.derivative)
print(k2.data, k2.derivative)
