import numpy as np
from scipy.special import gammaln
from tqdm import tqdm


class BootstrapPF:
    """
    Bootstrap particle filter (Gordon–Salmond–Smith) on h_t
    for the Poisson–Gamma Cox process.

    Parameters
    ----------
    nu, phi, c, beta : float
        Model parameters.
    N : int
        Number of particles.
    seed : int
        RNG seed.
    resample : bool
        Whether to resample at every time step.
    """

    def __init__(self, nu, phi, c, beta, N=20000, seed=1, resample=True):
        self.nu = float(nu)
        self.phi = float(phi)
        self.c = float(c)
        self.beta = float(beta)
        self.N = int(N)
        self.seed = int(seed)
        self.resample = bool(resample)

        self.rng = np.random.default_rng(self.seed)

    def run(self, y, progress=False):
        """
        Run the bootstrap particle filter.

        Parameters
        ----------
        y : array-like, shape (T,)
            Observed count time series.
        progress : bool
            Show a progress bar.

        Returns
        -------
        loglik : float
            PF estimate of log p(y_1:T).
        Eh : ndarray, shape (T,)
            Filtered mean E[h_t | y_1:t].
        """
        y = np.asarray(y, dtype=np.int64)
        T = len(y)

        # h0 ~ Gamma(nu, c/(1-phi))
        h = self.rng.gamma(
            shape=self.nu,
            scale=self.c / (1.0 - self.phi),
            size=self.N
        )

        loglik = 0.0
        Eh = np.zeros(T)

        iterator = range(T)
        if progress:
            iterator = tqdm(iterator, total=T,
                            desc=f"Bootstrap PF (N={self.N}, seed={self.seed})")

        for t in iterator:
            # propagate
            z = self.rng.poisson(lam=self.phi * h / self.c)
            h = self.rng.gamma(shape=self.nu + z, scale=self.c)

            # weights: y_t | h_t ~ Poisson(beta * h_t)
            lam = self.beta * h
            logw = y[t] * np.log(lam) - lam - gammaln(y[t] + 1.0)

            m = np.max(logw)
            w = np.exp(logw - m)
            loglik += np.log(np.mean(w)) + m

            w /= np.sum(w)
            Eh[t] = np.sum(w * h)

            if self.resample:
                idx = self.rng.choice(self.N, size=self.N, replace=True, p=w)
                h = h[idx]

        return loglik, Eh