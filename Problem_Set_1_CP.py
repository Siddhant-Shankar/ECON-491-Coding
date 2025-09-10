# A medical test is designed to detect a disease that about 3% of
# the population has. For 93% of those who have the disease, the test yields a positive result.
# In addition, the test falsely yields a positive result for 7% of those without the disease. What
# is the probability that a person has the disease given that they have tested positive? Answer
# this question by simulating data for 10,000 people.

import numpy as np
import pandas as pd


np.random.seed(42)

N = 10000 #num trials
prevalance = 0.03 #P (D = 1)
sensitivity = 0.93 #sensivity of the test -> P(+ | D = 1)
false_positive_rate = 0.07 #P(+ | D = 0)
specificity = 1 - false_positive_rate
#we need to find P(D=1 | +)

#Simulate diesease state: 
D = np.random.rand(N) < prevalance #generates a boolean array for disease
# Simulate test outcomes
test_positive = np.where(D,
                         np.random.rand(N) < sensitivity,      # if disease
                         np.random.rand(N) < false_positive_rate)  # if no disease

# Compute counts
tp = np.sum(test_positive & D)  # true positives
fp = np.sum(test_positive & ~D) # false positives
tn = np.sum((~test_positive) & (~D))  # true negatives
fn = np.sum((~test_positive) & D)     # false negatives

#Posterior Probablity Estimate
posterior_sim = tp/(tp + fp)

posterior_exact = (sensitivity * prevalance) / (sensitivity * prevalance + false_positive_rate * (1 - prevalance))

report = pd.DataFrame({
    "Metric": ["Population size", "Prevalance", "Sensitivity", "False positive rate", "Specificity",
               "True Positives", "False Positives", "True Negatives", "False Negatives",
               "P(D=1 | +) - simulated", "P(D=1 | +) - exact (Bayes)"],
    "Value": [N, prevalance, sensitivity, false_positive_rate, specificity,
              tp, fp, tn, fn, posterior_sim, posterior_exact]
})

print(report)
