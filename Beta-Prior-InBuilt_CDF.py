import numpy as np
from scipy.stats import beta

alpha, beta_param = 2, 5

# Points where we evaluate
x = np.linspace(0, 1, 5)

# PDF (probability density function)
pdf_vals = beta.pdf(x, a=alpha, b=beta_param)

# CDF (cumulative distribution function)
cdf_vals = beta.cdf(x, a=alpha, b=beta_param)

# Quantile function (inverse cdf)
q25 = beta.ppf(0.25, a=alpha, b=beta_param)  # 25th percentile

# Random draws
samples = beta.rvs(a=alpha, b=beta_param, size=10, random_state=0)

print("x =", x)
print("pdf =", pdf_vals)
print("cdf =", cdf_vals)
print("25th percentile =", q25)
print("random samples =", samples)
