import numpy as np
from scipy.special import gammaln
from tqdm import tqdm


class BootstrapPF:
    """Bootstrap particle filter for a Gamma-Poisson latent process with Poisson observations."""
    def __init__(self, nu, phi, c, expo, N=20000, seed=1, resample=True):
        self.nu = float(nu)
        self.phi = float(phi)
        self.c = float(c)
        self.expo = np.asarray(expo, dtype=float).ravel()
        self.N = int(N)
        self.rng = np.random.default_rng(seed)
        self.resample = bool(resample)

    def run(self, y, progress=False):
        y = np.asarray(y, dtype=np.int64)
        T = len(y)
        assert len(self.expo) == T

        h = self.rng.gamma(shape=self.nu, scale=self.c/(1.0-self.phi), size=self.N)

        loglik = 0.0
        Eh = np.zeros(T)

        it = range(T)
        if progress:
            it = tqdm(it, total=T, desc=f"Bootstrap PF (N={self.N})")

        for t in it:
            z = self.rng.poisson(lam=self.phi * h / self.c)
            h = self.rng.gamma(shape=self.nu + z, scale=self.c)

            lam = self.expo[t] * h
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
