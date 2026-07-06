"""
Q1) Implement numerical_second_derivative(f, x) using numerical_derivative called twice.
Verify that the second derivative of x^3 at x=2 is 12.
"""


def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)


def f(x):
    return x**3


def numerical_second_derivative(f, x, h=1e-4):
    second_derivative = numerical_derivative(
        lambda t: numerical_derivative(f, t, h),
        x,
        h,
    )
    return second_derivative


# print(numerical_derivative(f, 2))
print(numerical_second_derivative(f, 2))
