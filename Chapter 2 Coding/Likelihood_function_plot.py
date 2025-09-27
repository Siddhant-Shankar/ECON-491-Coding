import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# observation
n, y = 10, 7

# grid of possible p values between 0 and 1
p_grid = np.linspace(0, 1, 200)

# likelihood values: P(Y=7 | p)
likelihood = binom.pmf(y, n, p_grid)

# normalize so the maximum is 1 (optional, just for plotting nicely)
likelihood /= likelihood.max()

# plot
plt.plot(p_grid, likelihood, label="Likelihood for p given y=7")
plt.axvline(y/n, color="red", linestyle="--", label=f"MLE = {y/n:.2f}")
plt.xlabel("p")
plt.ylabel("Likelihood (scaled)")
plt.title("Binomial Likelihood, n=10, y=7")
plt.legend()
plt.show()