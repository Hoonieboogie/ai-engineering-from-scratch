"""
Q1) Implement numerical_second_derivative(f, x) using numerical_derivative called twice.
Verify that the second derivative of x^3 at x=2 is 12.
"""


def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)


def f(x):
    return x**3


def numerical_second_derivative(f, x, h=1e-4):

    def first_derivative_at(t):
        return numerical_derivative(f, t, h)

    # Lambda alternative:
    # return numerical_derivative(lambda t: numerical_derivative(f, t, h), x, h)
    return numerical_derivative(first_derivative_at, x, h)


# print(numerical_derivative(f, 2))
print(numerical_second_derivative(f, 2))

print("=" * 100)
"""
Q2) Use gradient descent to find the minimum of f(x, y) = (x - 3)^2 + (y + 1)^2.
Start from (0, 0). The answer should converge to (3, -1).
"""


def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        # Make two copies so the original point stays unchanged.
        point_plus = list(point)
        point_minus = list(point)
        # Move only the current variable a tiny step forward.
        point_plus[i] += h
        # Move only the current variable a tiny step backward.
        point_minus[i] -= h
        # Estimate the slope for this one variable using central difference.
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        # Save this partial derivative in the gradient vector.
        gradient.append(partial)
    return gradient


def f_multi(point):
    x, y = point
    return (x - 3) ** 2 + (y + 1) ** 2


point = [0.0, 0.0]
lr = 0.1
for step in range(30):
    grad = numerical_gradient(f_multi, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    loss = f_multi(point)
    if step % 5 == 0 or step == 29:
        print(f"step {step:2d}  point=({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")

print("=" * 100)
"""
Q3) Add momentum to the gradient descent loop:
maintain a velocity vector that accumulates past gradients.
Compare convergence speed with and without momentum on f(x) = x^4 - 3x^2.
"""


def f_Q3(x):
    return x**4 - 3 * (x**2)


velocity = 0.0  # past movement (0 for no past movement a step 0)
momentum = 0.5
point_q3 = 0.5
point_q3_mo = 0.5
lr_q3 = 0.01

for step in range(100):
    grad = numerical_derivative(f_Q3, point_q3)
    grad_mo = numerical_derivative(f_Q3, point_q3_mo)

    # without momentum
    point_q3 = point_q3 - lr_q3 * grad
    # with momentum
    velocity = momentum * velocity - lr_q3 * grad_mo
    point_q3_mo = point_q3_mo + velocity

    loss = f_Q3(point_q3)
    loss_mo = f_Q3(point_q3_mo)
    if step % 10 == 0 or step == 99:
        print(
            f"step {step:2d}  x=({point_q3:7.4f} | {point_q3_mo:7.4f})  f=({loss:.6f} | {loss_mo:.6f})"
        )
