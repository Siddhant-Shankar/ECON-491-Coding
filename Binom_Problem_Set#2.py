import math
import numpy as np
import matplotlib.pyplot as plt

# ---------- math helpers ----------
def B(a,b): return math.gamma(a)*math.gamma(b)/math.gamma(a+b)

def beta_pdf(x, a, b):
    if x <= 0 or x >= 1:
        return 0.0
    return (x**(a-1)) * ((1-x)**(b-1)) / B(a,b)

def beta_curve(a, b, grid=2000, eps=1e-6):
    xs = np.linspace(eps, 1-eps, grid)
    ys = np.array([beta_pdf(x, a, b) for x in xs])
    # clip spikes for aesthetics only
    cap = np.percentile(ys, 99.9)
    ys = np.minimum(ys, cap)
    return xs, ys

def sd_beta(a,b):
    return math.sqrt(a*b/((a+b)**2 * (a+b+1)))

def mode_beta(a,b):
    if a>1 and b>1:
        return (a-1)/(a+b-2)
    elif a<=1 and b>1:
        return 0.0
    elif b<=1 and a>1:
        return 1.0
    else:
        return float('nan')

def likelihood_curve(x, n, grid=2000):
    xs = np.linspace(1e-6, 1-1e-6, grid)
    ys = xs**x * (1-xs)**(n-x)
    # scale for plotting comparability
    ys /= ys.max()
    return xs, ys

# ---------- Ex 3.4 & 3.7 ----------
def plot_six_betas():
    order = [(0.5,6), (1,1), (6,2),
             (6,6), (0.5,0.5), (2,2)]
    titles = ["Beta(0.5,6)", "Beta(1,1)", "Beta(6,2)",
              "Beta(6,6)", "Beta(0.5,0.5)", "Beta(2,2)"]
    plt.figure(figsize=(11,6))
    for i,(a,b) in enumerate(order, start=1):
        xs, ys = beta_curve(a,b)
        ax = plt.subplot(2,3,i)
        ax.plot(xs, ys)
        ax.set_title(titles[i-1])
        ax.set_xlim(0,1)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("exercise_3_4_six_betas.png", dpi=220)
    plt.close()

# ---------- Ex 3.9 ----------
def priors_39():
    pairs = [(8,2,"ND Beta(8,2)"), (1,20,"LA Beta(1,20)")]
    # print numbers
    for a,b,lab in pairs:
        mean = a/(a+b)
        mode = mode_beta(a,b)
        sd   = sd_beta(a,b)
        print(f"{lab}: mean={mean:.6f}, mode={mode:.6f}, sd={sd:.6f}")
    # plot
    plt.figure(figsize=(7,4))
    for a,b,lab in pairs:
        xs, ys = beta_curve(a,b)
        plt.plot(xs, ys, label=lab)
    plt.xlim(0,1); plt.grid(alpha=0.2); plt.legend()
    plt.title("Exercise 3.9: Prior PDFs")
    plt.savefig("exercise_3_9_priors.png", dpi=220)
    plt.close()

# ---------- Ex 3.10 ----------
def tri_plot(prior_ab, x, n, fname, title):
    a0,b0 = prior_ab
    xs, prior = beta_curve(a0,b0)
    _, like  = likelihood_curve(x,n,grid=len(xs))
    _, post  = beta_curve(a0+x, b0+(n-x))
    # scale prior/post to max=1 for comparability
    prior /= prior.max()
    post  /= post.max()
    plt.figure(figsize=(8,4.5))
    plt.plot(xs, prior, label=f"Prior Beta({a0},{b0})")
    plt.plot(xs, like,  label=f"Likelihood Bin(n={n}, x={x}) (scaled)")
    plt.plot(xs, post,  label=f"Posterior Beta({a0+x},{b0+(n-x)})")
    plt.xlim(0,1); plt.grid(alpha=0.2); plt.legend()
    plt.title(title)
    plt.savefig(fname, dpi=220)
    plt.close()

def posteriors_310():
    n, x = 50, 12
    # ND
    a_nd, b_nd = 8, 2
    aN, bN = a_nd + x, b_nd + (n - x)
    meanN = aN/(aN+bN); modeN = mode_beta(aN,bN); sdN = sd_beta(aN,bN)
    print(f"ND posterior Beta({aN},{bN}): mean={meanN:.6f}, mode={modeN:.6f}, sd={sdN:.6f}")
    tri_plot((a_nd,b_nd), x, n, "exercise_3_10_ND.png", "Ex 3.10: ND (prior, likelihood, posterior)")

    # LA
    a_la, b_la = 1, 20
    aL, bL = a_la + x, b_la + (n - x)
    meanL = aL/(aL+bL); modeL = mode_beta(aL,bL); sdL = sd_beta(aL,bL)
    print(f"LA posterior Beta({aL},{bL}): mean={meanL:.6f}, mode={modeL:.6f}, sd={sdL:.6f}")
    tri_plot((a_la,b_la), x, n, "exercise_3_10_LA.png", "Ex 3.10: LA (prior, likelihood, posterior)")

# ---------- Ex 3.11 ----------
def bikes_311():
    # Solve already known: Beta(6,18)
    xs, ys = beta_curve(6,18)
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys); plt.grid(alpha=0.2)
    plt.title("Ex 3.11(a): Prior Beta(6,18)")
    plt.xlim(0,1)
    plt.savefig("exercise_3_11_prior.png", dpi=220)
    plt.close()

    # Posterior with n=50, x=15 => Beta(21,53)
    a,b = 21,53
    mean = a/(a+b)
    mode = mode_beta(a,b)
    sd   = sd_beta(a,b)
    print(f"Bike posterior Beta({a},{b}): mean={mean:.6f}, mode={mode:.6f}, sd={sd:.6f}")

    xs, ys = beta_curve(a,b)
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys); plt.grid(alpha=0.2)
    plt.title("Ex 3.11(b): Posterior Beta(21,53)")
    plt.xlim(0,1)
    plt.savefig("exercise_3_11_posterior.png", dpi=220)
    plt.close()

# ---------- main ----------
if __name__ == "__main__":
    plot_six_betas()   # Ex 3.4 & 3.7(a)
    priors_39()        # Ex 3.9(a,b) + prints numbers
    posteriors_310()   # Ex 3.10(a,b,c) + prints numbers
    bikes_311()        # Ex 3.11(a,b,c)
