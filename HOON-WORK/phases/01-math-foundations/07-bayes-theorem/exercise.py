"""
Q1) Multiple tests.
A patient tests positive twice on independent tests (both 99% accurate,
disease prevalence 1 in 10,000).
What is P(sick) after both tests?
Use the posterior from the first test as the prior for the second.
"""

import bayes

prior = 0.0001  # P(sick)
likelihood = 0.99  # P(positive | sick)
posterior_1 = bayes.bayes(prior, likelihood, 1 - likelihood)
posterior_2 = bayes.bayes(posterior_1, likelihood, 1 - likelihood)
sequential_posterior = bayes.sequential_bayes(prior, likelihood, 1 - likelihood, 2)

if abs(posterior_2 - sequential_posterior) < 1e-12:
    print(f"Both calculations match: {posterior_2:.1%}")
else:
    print("The calculations do not match.")

print("-----------------------------------------------------")
"""
Q2) Smoothing impact.
Run the spam classifier with smoothing values of 0.01, 0.1, 1.0, and 10.0.
How do the top word probabilities change?
What happens with smoothing=0 and a word that appears only in ham?
"""


"""
Q3) Add features.
Extend the NaiveBayes class to also use message length (short/long) as a feature alongside word counts.
Estimate P(short|spam) and P(short|ham) from the training data and fold it into the prediction score.
"""


"""
Q4) MAP by hand.
Given observed data (7 heads in 10 coin flips),
compute the MAP estimate of the bias using a Beta(2,2) prior.
Compare it to the MLE estimate (7/10).
"""
