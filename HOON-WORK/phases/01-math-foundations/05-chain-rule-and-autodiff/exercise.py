"""
Q1) Add __pow__ to the Value class so you can compute x ** n.
Verify that d/dx(x^3) at x=2 equals 12.0.
"""
class Value:
    """A number plus the information needed to calculate derivatives later.

    Vocabulary used throughout this class:
    - data: the ordinary numeric value produced in the forward pass.
    - grad: how much the final answer changes when this Value changes.
    - _prev: the earlier Value objects used to create this Value.
    - _backward: the small chain-rule instruction for sending grad to _prev.
    """

    def __init__(self, data, children=(), op=""):
        # Store the ordinary number, for example 2.0 or the result of a + b.
        self.data = float(data)

        # No gradient is known at first. backward() calculates it later.
        self.grad = 0.0

        # Leaf inputs have no earlier operation, so their default backward
        # instruction does nothing. Operations replace this with a real rule.
        self._backward = lambda: None

        # Keep the Values that created this Value. This is the graph history.
        self._prev = set(children)

        # Keep a human-readable label such as "+" or "*" for inspection only.
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        # This method runs when Python sees: a + b.
        # Here, self is a and other is b.
        # If b is a plain number, such as 3, turn it into Value(3) first.
        # Then both a and b have the same useful fields: .data and .grad.
        other = other if isinstance(other, Value) else Value(other)

        # FORWARD PASS: do the ordinary arithmetic right now.
        # The new object out holds the answer a + b in out.data.
        # (self, other) means: "out was made from these two earlier Values."
        # "+" is only a label that helps us inspect the graph.
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # This code does NOT run now. It runs later, during out.backward().
            # WHY is d(out)/d(a) = 1 for out = a + b?
            # Hold b fixed. Change a by a small amount Δa:
            #   old out = a + b
            #   new out = (a + Δa) + b
            #   new out - old out = Δa
            # The output changes by exactly the same amount as a, so
            #   Δout / Δa = 1  and therefore d(out)/d(a) = 1.
            # The identical argument, while holding a fixed, gives
            #   d(out)/d(b) = 1.
            # out.grad means: "how much does the FINAL answer depend on out?"
            # So that same amount is passed to both a and b.
            # We use +=, not =, because a can appear in more than one place.
            # Example: if y = a + a, both paths must contribute to a.grad.
            self.grad += out.grad
            other.grad += out.grad

        # Attach this saved instruction to out. The main backward() method
        # will find out and call this instruction at the correct time.
        out._backward = _backward

        # Give the new result back to the code that wrote a + b.
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        # This method runs when Python sees: a * b.
        # Again, wrap b if it is a plain number such as 3.
        other = other if isinstance(other, Value) else Value(other)

        # FORWARD PASS: calculate the normal product now.
        # Remember that this new result came from self and other.
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # This code runs later during out.backward().
            # For out = a * b:
            # WHY is d(out)/d(a) = b? Hold b fixed and change a by Δa:
            #   old out = a * b
            #   new out = (a + Δa) * b = a*b + Δa*b
            #   new out - old out = Δa*b
            # Dividing by Δa gives Δout/Δa = b, so d(out)/d(a) = b.
            # By the same reasoning, while holding a fixed:
            #   d(out)/d(b) = a.
            # out.grad tells us how much the FINAL answer depends on out.
            # The chain rule says, for example:
            #   d(final)/d(a) = d(final)/d(out) * d(out)/d(a)
            #                  = out.grad          * b.data
            # += again keeps contributions from every possible graph path.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        # Save multiplication's special backward instruction on out.
        out._backward = _backward

        # Give back the new result object for a * b.
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
        # This method runs when Python sees: x ** n.
        # Here, self is x and n is a plain exponent such as 2 or 3.

        # FORWARD PASS: calculate x raised to the power n right now.
        # There is one parent, self, because x ** n depends only on x.
        # The label, for example "**3", is only for inspecting the graph.
        out = Value(self.data**n, (self,), f"**{n}")

        def _backward():
            # This code runs later during out.backward().
            # The calculus power rule says:
            #   d(x ** n)/dx = n * (x ** (n - 1))
            # Example: d(x ** 3)/dx = 3 * (x ** 2).
            #
            # out.grad means: "how much does the FINAL answer depend on out?"
            # Apply the chain rule:
            #   d(final)/d(x) = d(final)/d(out) * d(out)/d(x)
            #                 = out.grad * n * (x ** (n - 1))
            # += keeps this contribution if x is also used elsewhere.
            self.grad += n * (self.data ** (n - 1)) * out.grad

        # Save this power-specific backward instruction on the new output.
        out._backward = _backward

        # Give back the new Value created by x ** n.
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
        # You call this on the final answer, for example: y.backward().
        # Its job is to apply the chain rule through the ENTIRE graph.
        # We must visit every Value that helped make y, in a safe order.
        # A node can only pass its gradient to earlier nodes after it has
        # received every gradient contribution from later nodes.
        # topo will first store them from earliest input to final output.
        topo = []

        # A Value may be reused, for example y = a + a.
        # visited remembers nodes we already saw, so we do not process twice.
        visited = set()

        def build_topo(v):
            # v is one Value node in the calculation graph.
            # Work on it only if we have not already seen it.
            if v not in visited:
                visited.add(v)

                # v._prev contains the earlier Values used to create v.
                # Visit those earlier Values before adding v itself.
                for child in v._prev:
                    build_topo(child)

                # Now all of v's inputs are already in topo, so add v.
                topo.append(v)

        # Start at the final answer (self) and collect its full history.
        build_topo(self)

        # Start the backward pass with: d(final answer)/d(final answer) = 1.
        # In plain language: if y changes by Δy, then y changes by Δy.
        # This is the first known gradient. Every earlier gradient is found
        # by multiplying this value through one local derivative at a time.
        self.grad = 1.0

        # topo currently goes inputs -> output. Reverse it to go output -> inputs.
        # For each node, run the small saved rule from +, *, relu, and so on.
        # Each small rule uses this chain-rule pattern:
        #   d(final)/d(parent) = d(final)/d(node) * d(node)/d(parent)
        # In plain language: "effect on final answer" times "local effect."
        # Each small rule then passes its gradient backward to that node's inputs.
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
k1 = x + c
k2 = x * c
print(k1.data, k1.derivative)
print(k2.data, k2.derivative)

x_ = Value(2)
c_ = Value(3)
k1_ = x_ + c_
k2_ = x_ * c_

# Backward flow for k1_ = x_ + c_:
# 1. k1_.backward() seeds dk1/dk1 = 1, so k1_.grad becomes 1.
# 2. Addition sends that gradient unchanged to both inputs:
#    dk1/dx = 1 and dk1/dc = 1.
# 3. The Value class accumulates those contributions in x_.grad and c_.grad.
#
# Backward flow for k2_ = x_ * c_:
# 1. k2_.backward() seeds dk2/dk2 = 1.
# 2. Multiplication sends dk2/dx = c_ and dk2/dc = x_.
# 3. Gradients use +=, so calling backward on both k1_ and k2_ accumulates
#    their contributions in the shared x_.grad and c_.grad values.

k1_.backward()
print("comparison) x_.grad:", x_.grad, "k1.grad:", k1.derivative)
