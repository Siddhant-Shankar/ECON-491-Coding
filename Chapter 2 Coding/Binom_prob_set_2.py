import numpy as np
from scipy.stats import beta, binom
import matplotlib.pyplot as plt

# # observation
# x = np.linspace(0, 1, 200)
# params = [(1,1), (2,2), (0.5,0.5), (6,6)]

# for a,b in params:
#     plt.subplot(2,2,params.index((a,b))+1)
#     plt.plot(x, beta.pdf(x, a, b), label=f'Beta({a},{b})')
#     plt.title(f'Prior Beta({a},{b})')
#     plt.xlabel('x')
#     plt.ylabel('Density (scaled)')
#     plt.ylim(0, 1.1)
#     plt.grid(alpha=0.2)
#     plt.legend()
# plt.tight_layout()
# plt.show()


# x = np.linspace(0, 1, 100)
# params = [(8,2) , (1, 20)]
# plt.figure(figsize=(8, 4))
# for a,b in params: 
#     plt.plot(x, beta.pdf(x, a, b), label=f'Beta({a},{b})')

# plt.legend()
# plt.title('Exercise 3.9: Prior PDFs')
# plt.xlabel('x')
# plt.ylabel('Density (scaled)')
# plt.show()

x = np.linspace(0, 1, 200)
n,y = 50,12
a, b = 8, 2
post_a, post_b = 20, 38
prior = beta.pdf(x, a, b)
like = binom.pmf(y, n, x)
post = beta.pdf(x, post_a, post_b)

plt.figure(figsize=(8, 5))
plt.plot(x, prior/prior.max(), label=f'Prior Beta({a},{b})')
plt.plot(x, like/like.max(), label=f'Likelihood Bin(n={n}, y={y}) (scaled)')
plt.plot(x, post/post.max(), label=f'Posterior Beta({post_a},{post_b})')
plt.title('Exercise 3.10: Prior, Likelihood, Posterior')
plt.xlabel('x')
plt.ylabel('Density (scaled)')
plt.legend()
plt.show()

