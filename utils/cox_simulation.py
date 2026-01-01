import numpy as np


class CoxProcessSimulation:
    """
    Cox process (Creal 2017, Sec. 2.2.1 / 5.1).
    """

    def __init__(self, T, phi, nu, c, beta=1.0, seed=None):
        if not (T > 0):
            raise ValueError("T must be > 0")
        if not (0.0 < phi < 1.0):
            raise ValueError("phi must be in (0,1)")
        if nu <= 0 or c <= 0:
            raise ValueError("nu and c must be > 0")
        if beta <= 0:
            raise ValueError("beta must be > 0")

        self.T = int(T)
        self.phi = float(phi)
        self.nu = float(nu)
        self.c = float(c)
        self.beta = float(beta)
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

        # --- initial state h_0 ---
        if init == "stationary":
            burn = 0
            total = self.T
            h_prev = self.rng.gamma(
                shape=self.nu,
                scale=self.c / (1.0 - self.phi)
            )
        else:
            burn = int(burn_in)
            total = self.T + burn
            h_prev = float(h0)

        # storage for t = 1..total  (index 0..total-1)
        h = np.zeros(total)
        z = np.zeros(total, dtype=int)
        y = np.zeros(total, dtype=int)
        lam = np.zeros(total) if return_lambda else None

        for t in range(total):
            # z_t | h_{t-1}
            lam_t = (self.phi * h_prev) / self.c
            if return_lambda:
                lam[t] = lam_t

            z[t] = self.rng.poisson(lam_t)

            # h_t | z_t
            h[t] = self.rng.gamma(
                shape=self.nu + z[t],
                scale=self.c
            )

            # y_t | h_t
            y[t] = self.rng.poisson(
                lam=self.beta * h[t]
            )

            h_prev = h[t]

        # drop burn-in if needed
        if burn > 0:
            h = h[burn:]
            z = z[burn:]
            y = y[burn:]
            if return_lambda:
                lam = lam[burn:]

        return (y, h, z, lam) if return_lambda else (y, h, z)