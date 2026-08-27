import math
import optimizers

"""
Q1) Learning rate sweep.
Run vanilla gradient descent on the Rosenbrock function
with learning rates [0.0001, 0.0005, 0.001, 0.005, 0.01].
Plot or print the final loss after 5000 steps for each.
Find the largest learning rate that still converges.
"""
print("<(Q1) Learning Rate Effect on Gradient Descent>")

start = (-1.0, 1.0)  # (x,y)
lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01]
steps = 5000
stable_lrs = []

for lr in lrs:
    gd = optimizers.GradientDescent(lr)
    history = optimizers.optimize(
        gd, optimizers.rosenbrock, optimizers.rosenbrock_gradient, start, steps
    )
    initial_loss = optimizers.rosenbrock(history[0])
    final_loss = optimizers.rosenbrock(history[-1])

    # A stable run completes every requested step and reduces the loss.
    stable = (
        len(history) == steps + 1  # Did it finish all 5,000 steps?
        and math.isfinite(final_loss)  # Is the final loss a normal number?
        and final_loss < initial_loss  # Did the loss go down?
    )
    if stable:
        stable_lrs.append(lr)

    status = "stable" if stable else "diverged"
    print(f"lr={lr} initial={initial_loss:.8g} final={final_loss:.8g} ({status})")

if stable_lrs:
    print(f"Largest stable learning rate: {max(stable_lrs)}")
else:
    print("No learning rate completed the sweep with lower final loss.")


"""
Q2) Momentum comparison.
Run SGD with momentum values [0.0, 0.5, 0.9, 0.99] on the Rosenbrock function.
Track the loss at every step.
Which momentum value converges fastest? Which overshoots?
"""
print("=" * 100)
print("<(Q2) Momentum Effect on SGD>")
lr = 0.0001
momentums = [0.0, 0.5, 0.9, 0.99]
start = (-1.0, 1.0)
steps = 5000

convergence_steps = {}
overshoot = {}
for beta in momentums:
    sgd = optimizers.SGDMomentum(lr, beta)
    history = optimizers.optimize(
        sgd, optimizers.rosenbrock, optimizers.rosenbrock_gradient, start, steps
    )  # history = [[x1,y1], [x2,y2], ...]
    losses = [optimizers.rosenbrock(params) for params in history]
    threshold = 1e-4

    convergence_step = next(
        (step for step, loss in enumerate(losses) if loss < threshold),
        None,
    )  # first loss below the threshold
    convergence_steps[beta] = convergence_step

    overshoot_steps = [
        step for step in range(1, len(losses)) if losses[step] > losses[step - 1]
    ]
    if len(overshoot_steps) > 0:
        overshoot[beta] = overshoot_steps

# Exclude beta that never reached the threshold
reached_threshold = {
    beta: step for beta, step in convergence_steps.items() if step is not None
}

fastest_beta = min(reached_threshold, key=reached_threshold.get)
print(
    f"Beta with the fastest convergence: {fastest_beta}"
    f" at step {convergence_steps[fastest_beta]}"
)

print(f"Beta with overshoot: {list(overshoot)}")
# How much oscillates
for beta, steps_with_overshoot in overshoot.items():
    print(f"beta={beta}: {len(steps_with_overshoot)} overshoot steps")


"""
Q3) Saddle point escape.
Define the function f(x, y) = x^2 - y^2 (a saddle point at the origin).
Start at (0.01, 0.01).
Compare how vanilla GD, SGD with momentum, and Adam behave.
Which escapes the saddle point?
"""
print("=" * 100)
print("<(Q3) Saddle point escape>")


def saddle(params):
    x, y = params
    return x**2 - y**2


def saddle_gradient(params):
    x, y = params
    df_dx = 2 * x
    df_dy = -2 * y
    return [df_dx, df_dy]


start = [0.01, 0.01]  # x,y
optimizers_list = {
    "GD": optimizers.GradientDescent(lr=0.01),
    "SGDM": optimizers.SGDMomentum(lr=0.01, momentum=0.9),
    "ADAM": optimizers.Adam(lr=0.01),
}
escaped = []
for name, opt in optimizers_list.items():
    history = optimizers.optimize(opt, saddle, saddle_gradient, start, 200)
    last = history[-1]
    if abs(last[1]) > 1:
        escaped.append(name)
    print(
        f"{name}: x={last[0]:.6f}, y={last[1]:.6f}, "
        f"loss={saddle(last):.6f}, escaped={abs(last[1]) > 1}"
    )

# Escaped
print(escaped)


"""
Q4) Implement learning rate decay.
Add an exponential decay schedule to the GradientDescent class: lr = lr_0 * 0.999^step.
Compare convergence with and without decay on the Rosenbrock function.
"""
print("=" * 100)
print("<(Q4) Learning Decay schedule>")


# Learning decay schedule added
class GradientDescentDecay:
    def __init__(self, lr=0.001):
        self.lr = lr
        self.t = 0

    def step(self, params, grads):
        current_lr = self.lr * (0.999**self.t)
        self.t += 1
        return [p - current_lr * g for p, g in zip(params, grads)]


optimizer_list = [optimizers.GradientDescent(), GradientDescentDecay()]
start = [-1.0, 1.0]
for opt in optimizer_list:
    history = optimizers.optimize(
        opt, optimizers.rosenbrock, optimizers.rosenbrock_gradient, start, steps=600
    )

    for i in range(len(history)):
        if i % 100 == 0:
            print(f"step{i}: {round(optimizers.rosenbrock(history[i]), 5)} |", end=" ")
    print()
