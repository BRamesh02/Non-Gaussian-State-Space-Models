import numpy as np


class CoxProcessSimulation:
    """
    Cox process (Creal 2017, Sec. 2.2.1).
    """

    def __init__(self, T, phi, nu, c, seed=None):
        if not (T > 0):
            raise ValueError("T must be > 0")
        if not (0.0 < phi < 1.0):
            raise ValueError("phi must be in (0,1)")
        if nu <= 0 or c <= 0:
            raise ValueError("nu and c must be > 0")

        self.T = int(T)
        self.phi = float(phi)
        self.nu = float(nu)
        self.c = float(c)
        self.rng = np.random.default_rng(seed)

    def simulate(self, init="stationary", burn_in=500, h0=1.0, return_lambda=False):
        """
        Parameters
        ----------
        init : {"stationary", "burnin"}
            - "stationary": h0 ~ Gamma(nu, c/(1-phi)), simulate T
            - "burnin": start at h0, simulate burn_in + T, keep last T
        burn_in : int
            Length of burn-in if init="burnin"
        h0 : float
            Initial value if init="burnin"
        return_lambda : bool
            If True, also return lambda_t = phi * h_{t-1} / c

        Returns
        -------
        y : ndarray (T,)
        h : ndarray (T,)
        z : ndarray (T,)
        lam : ndarray (T,), optional
        """

        if init not in {"stationary", "burnin"}:
            raise ValueError("init must be 'stationary' or 'burnin'")

        if init == "stationary":
            total = self.T
            burn = 0
            h_init = self.rng.gamma(
                shape=self.nu,
                scale=self.c / (1.0 - self.phi)
            )
        else:
            burn = int(burn_in)
            total = self.T + burn
            h_init = float(h0)

        h = np.zeros(total)
        z = np.zeros(total, dtype=int)
        y = np.zeros(total, dtype=int)
        lam = np.zeros(total) if return_lambda else None

        h[0] = h_init
        y[0] = self.rng.poisson(h[0])
        z[0] = 0
        if return_lambda:
            lam[0] = np.nan  # non défini à t=0

        for t in range(1, total):
            lam_t = (self.phi * h[t - 1]) / self.c
            if return_lambda:
                lam[t] = lam_t

            z[t] = self.rng.poisson(lam_t)
            h[t] = self.rng.gamma(shape=self.nu + z[t], scale=self.c)
            y[t] = self.rng.poisson(h[t])

        if burn > 0:
            h = h[burn:]
            z = z[burn:]
            y = y[burn:]
            if return_lambda:
                lam = lam[burn:]

        return (y, h, z, lam) if return_lambda else (y, h, z)

