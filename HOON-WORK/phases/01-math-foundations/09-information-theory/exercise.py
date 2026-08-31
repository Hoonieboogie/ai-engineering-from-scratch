import information_theory as infth
import math

"""
Q1) Compute the entropy of the English alphabet assuming uniform distribution (26 letters).
Then estimate it using actual letter frequencies.
Which is higher and why?
"""
print("<Q1: Entropy Intro>")

alphabet = list("abcdefghijklmnopqrstuvwxyz")
uniform_distribution = [1 / len(alphabet) for _ in range(len(alphabet))]

e = math.e
uniform_entropy = infth.entropy(uniform_distribution, e)

corpus = "Machine learning systems make predictions from patterns in data. A model becomes useful when it assigns high probability to likely outcomes and low probability to unlikely ones. Careful evaluation helps us understand uncertainty, compare predictions, and improve decisions."

# corpus normalization
normalized = "".join(ch.lower() for ch in corpus if "a" <= ch.lower() <= "z")
alphabet_cnt = {}
for a in normalized:
    if a in alphabet_cnt:
        alphabet_cnt[a] += 1
    else:
        alphabet_cnt[a] = 1

for c in alphabet:
    if c not in alphabet_cnt:
        alphabet_cnt[c] = 0

alphabet_cnt = dict(sorted(alphabet_cnt.items()))

distribution = [cnt / sum(alphabet_cnt.values()) for cnt in alphabet_cnt.values()]
normalized_entropy = infth.entropy(distribution, e)
print(
    f"Comparison: {round(uniform_entropy, 3)}(uniform distribution) vs. {round(normalized_entropy,3)}(corpus-based distributio)"
)


"""
Q2) A model outputs logits [5.0, 2.0, 0.5] for a sample with true class 1.
Compute the cross-entropy loss by hand, then verify with your cross_entropy_loss function.
What logits would give zero loss?
"""
print("=" * 100)
print("<Q2: Cross-Entropy>")

# Logit --> Probability
logits = [5.0, 2.0, 0.5]
exp_logits = [math.exp(logit) for logit in logits]
summed_exp = sum(exp_logits)

model_probs = []
for exp_logit in exp_logits:
    prob = exp_logit / summed_exp
    model_probs.append(prob)

## verification
print(sum(model_probs))

# Cross-Entropy
## H(P, Q) = - ∑ p(x)*log(q(x)) for all x
## For classification, true probability is one hot-vector ==> H(P,Q) = -log(Q) *base e
true_probs = [0.0, 1.0, 0.0]
log_model_probs = [math.log(prob) for prob in model_probs]

h = [p * log_q for p, log_q in zip(true_probs, log_model_probs)]
H = -sum(h)

# Given Cross-Entropy
H_comp = infth.cross_entropy(true_probs, model_probs, base=math.e)

print(f"Cross-Entropy(mine): {H} | Cross-Entrop(given): {H_comp}")

# What logits would give zero loss?
## Let i be the true-class index. For a one-hot target:
##   L = -ln(q_i)
## Therefore, L = 0 would require q_i = 1.
##
## Softmax converts logits z into the true-class probability:
##   q_i = e^(z_i) / (e^(z_i) + ∑ e^(z_j) for every j != i)
##
## Divide the numerator and denominator by e^(z_i):
##   q_i = 1 / (1 + ∑ e^(z_j - z_i) for every j != i)
##
## For finite logits, every e^(z_j - z_i) is positive, so the denominator
## is greater than 1 and q_i is strictly less than 1. The denominator is
## not supposed to approach zero; the competing terms must become negligible.
##
## Thus we need z_i - z_j -> +infinity for every competing class j.
## For example, with class i fixed as the true class, [0, K, 0] approaches
## zero loss as K -> +infinity, but no finite logit vector gives exactly 0.
## (Floating-point software may display 0 after rounding.)
