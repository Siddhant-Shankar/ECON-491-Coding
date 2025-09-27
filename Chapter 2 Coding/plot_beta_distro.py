import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 500)
params = [(2,2), (10,2), (1, 1), (0.5, 0.5)]
plt.figure(figsize=(8, 6))
for a,b in params: 
    plt.plot(x, beta.pdf(x, a, b), label=f'Beta({a},{b})')

plt.legend()

plt.title('Beta Distribution')
plt.show()
