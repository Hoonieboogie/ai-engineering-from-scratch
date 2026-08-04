"""
Q1) Implement inverse transform sampling for the exponential distribution.
Verify by sampling 10,000 values and comparing the histogram to the true PDF.
"""
# The exponential distribution models a nonnegative waiting time, x >= 0.
# Use "rate" as the Python variable name: "lambda" is a reserved Python keyword.
#
# PDF (probability density function):
#     f(x) = rate * exp(-rate * x)
# CDF (cumulative distribution function):
#     F(x) = P(X <= x)
#          = integral from 0 to x of rate * exp(-rate * t) dt
#          = 1 - exp(-rate * x) --> 1 = F(x) + exp(-rate * x)
# The CDF is the probability that the waiting time has finished by x.
#
# Sanity checks:
#     F(0) = 0: no positive waiting time has finished before time 0.
#     As x becomes very large, F(x) approaches 1.
#
# exp(-rate * x) is instead the survival function P(X > x):
# the probability that the waiting time has NOT finished yet by x.
# Therefore: CDF + survival = 1.
import random
import math
from pathlib import Path

# Inverse of exponential CDF
# 1: F(X) = U = 1 - e^(-rate * X)
# 2: e^(-rate * X) = 1 - U
# 3: -rate * X = log(1 - U)
# 4: X = -log(1 - U) / rate
def inverse_exp_cdf_dist(rate, U):
    X = - math.log(1 - U) / rate
    return X


def exponential_pdf(rate, x):
    return rate * math.exp(-rate * x)


# Draw an empirical histogram of inverse-transform samples and overlay the
# theoretical PDF so the sampled distribution can be compared to its target.
import matplotlib.pyplot as plt

if __name__ == "__main__":
    rate = 1.5
    sample_count = 10_000
    samples = [inverse_exp_cdf_dist(rate, random.random()) for _ in range(sample_count)]

    # The x-axis represents waiting-time values, not generation order.
    max_x = max(samples)
    xs = [max_x * index / 200 for index in range(201)]
    pdf_values = [exponential_pdf(rate, x) for x in xs]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(
        samples,
        bins=30,
        density=True,
        alpha=0.65,
        label="10,000 inverse-transform samples",
    )
    axis.plot(xs, pdf_values, color="crimson", linewidth=2.5, label="True exponential PDF")
    axis.set(
        title=f"Exponential samples from inverse transform (rate = {rate})",
        xlabel="Waiting time x",
        ylabel="Probability density",
        xlim=(0, max_x),
    )
    axis.legend()
    figure.tight_layout()

    output_path = Path(__file__).with_name("inverse_exp_dist_cdf.png")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    #print(f"Saved histogram and PDF comparison: {output_path}")


"""
Q2) Build a joint distribution table for two loaded dice. 
Compute the marginal distributions and check whether the dice are independent.
"""
def build_independent_joint_distribution(dice_1_probs, dice_2_probs):
    """Construct a joint table while explicitly assuming independent dice."""
    return [
        [dice_1_prob * dice_2_prob for dice_2_prob in dice_2_probs]
        for dice_1_prob in dice_1_probs
    ]


def marginal_distributions(joint_dist):
    """Sum rows and columns to recover each die's individual probabilities."""
    dice_1_marginal = [sum(row) for row in joint_dist]
    dice_2_marginal = [
        sum(row[column] for row in joint_dist)
        for column in range(len(joint_dist[0]))
    ]
    return dice_1_marginal, dice_2_marginal


def are_independent(joint_dist, dice_1_marginal, dice_2_marginal, tolerance=1e-9):
    """Return True only when every joint cell equals the product of marginals."""
    # Independence requires P(A=i, B=j) = P(A=i) * P(B=j) for every cell.
    # math.isclose handles tiny floating-point rounding differences safely.
    return all(
        math.isclose(
            joint_dist[row][column],
            dice_1_marginal[row] * dice_2_marginal[column],
            abs_tol=tolerance,
        )
        for row in range(len(joint_dist))
        for column in range(len(joint_dist[row]))
    )


def print_joint_distribution(joint_dist):
    """Print a labeled table; rows are Die 1 faces and columns are Die 2 faces."""
    face_labels = [f"B={face}" for face in range(1, len(joint_dist[0]) + 1)]
    print("       " + " ".join(f"{label:>7}" for label in face_labels))
    for face, row in enumerate(joint_dist, start=1):
        values = " ".join(f"{probability:7.3f}" for probability in row)
        print(f"A={face}  {values}")


if __name__ == "__main__":
    dice_1_probs = [0.1, 0.2, 0.3, 0.1, 0.2, 0.1]
    dice_2_probs = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]

    joint_dist = build_independent_joint_distribution(dice_1_probs, dice_2_probs)
    marginal_dice_1, marginal_dice_2 = marginal_distributions(joint_dist)

    print("Joint distribution table (constructed assuming independence)")
    print_joint_distribution(joint_dist)
    print(f"\nDie 1 marginal: {[round(probability, 3) for probability in marginal_dice_1]}")
    print(f"Die 2 marginal: {[round(probability, 3) for probability in marginal_dice_2]}")
    print(
        "Independent by construction?: "
        f"{are_independent(joint_dist, marginal_dice_1, marginal_dice_2)}"
    )

