import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 500)
plt.plot(x, norm.pdf(x, loc=0, scale=1), label="μ=0, σ=1") #loc is mean, scale is stdev
plt.plot(x, norm.pdf(x, loc=2, scale=1), label="μ=2, σ=1")
plt.plot(x, norm.pdf(x, loc=-1, scale=2), label="μ=-1, σ=2")
plt.legend(); plt.title("Normal Distributions"); plt.show()