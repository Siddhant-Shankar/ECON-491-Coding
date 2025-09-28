import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
muTrue, sigmaTrue = 1, 0.5
n = 40
y = np.random.normal(muTrue, sigmaTrue, n)
print(y)
print(np.mean(y))
muMle = np.mean(y)
sigmaMLE = np.mean((y - muMle)**2)
print("MLE for mu:", muMle, " MLE for sigma^2:", sigmaMLE)
muSeq = np.linspace(-1, 3, 200)
logLike = []
for mu in muSeq: 
    logLike.append(np.sum(norm.logpdf(y, mu, sigmaTrue)))

plt.plot(muSeq, logLike)
plt.axvline(np.mean(y), color='red', linestyle='--', label='True mu')
plt.title('Log-Likelihood for Normal Distribution')
plt.legend()
plt.show()
