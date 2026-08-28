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
