from scipy.stats import binom, norm, geom

print(binom.pmf(3, 10, 0.5)) #P(X = 3) with n = 10, p = 0.5;
print(binom.cdf(3, 10, 0.5)) #P(X ≤ 3) with n = 10, p = 0.5;
print(binom.ppf(0.118, 10, 0.5)) #PPF is the inverse of CDF, it gives the smallest value k such that P(X ≤ k) ≥ q.

print(norm.pdf(0, loc=0, scale=1)) #Probability Density Function
print(norm.cdf(1.96)) #Cumulative distribution function
