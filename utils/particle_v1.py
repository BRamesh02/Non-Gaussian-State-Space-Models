import numpy as np
from math import lgamma


def log_poisson_pmf(y, lam):
    # log P(Y=y | lambda=lam) = y log lam - lam - log(y!)
    if lam <= 0.0:
        return -np.inf
    return y * np.log(lam) - lam - lgamma(y + 1.0)


def bootstrap_pf_counts(y, phi, nu, c, N=5000, seed=123):
    """
    Bootstrap particle filter on h_t:
      h0 ~ Gamma(nu, c/(1-phi))   ("stationary-ish")
      zt | h_{t-1} ~ Poisson(phi*h_{t-1}/c)
      ht | zt ~ Gamma(nu+zt, c)
      yt | ht ~ Poisson(ht)

    Resampling: multinomial every step.

    Returns
    -------
    loglik_hat : float
    filtered_mean_h : np.ndarray, shape (T,)
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int64)
    T = y.size

    # init particles
    h_particles = rng.gamma(shape=nu, scale=c / (1.0 - phi), size=N)
    w = np.ones(N, dtype=float) / N

    loglik = 0.0
    filt_mean = np.zeros(T, dtype=float)

    for t in range(T):
        if t > 0:
            # propagate: sample z_t | h_{t-1}, then h_t | z_t
            lam_z = (phi * h_particles) / c
            z = rng.poisson(lam_z)
            h_particles = rng.gamma(shape=nu + z, scale=c)

        # weight with Poisson(y_t | h_t)
        yt = int(y[t])
        logw = np.array([log_poisson_pmf(yt, lam) for lam in h_particles], dtype=float)

        # stable normalization + incremental loglik
        m = np.max(logw)
        w_unnorm = np.exp(logw - m)
        s = np.sum(w_unnorm)

        w = w_unnorm / s
        loglik += (m + np.log(s) - np.log(N))

        # filtered mean
        filt_mean[t] = np.sum(w * h_particles)

        # resample (multinomial)
        idx = rng.choice(N, size=N, replace=True, p=w)
        h_particles = h_particles[idx]
        w.fill(1.0 / N)

    return loglik, filt_mean


def loglik_pf_once(y, phi, nu, c, N=2000, seed=0):
    ll, _ = bootstrap_pf_counts(y, phi=phi, nu=nu, c=c, N=N, seed=seed)
    return ll
