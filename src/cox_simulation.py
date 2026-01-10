import numpy as np


class CoxProcessSimulation:
    """
    Cox process with frailty (Creal 2017, Sec. 2.2.1 / 5.1).

    Latent frailty dynamics (Gamma–Poisson–Gamma):
        z_t | h_{t-1} ~ Poisson(phi * h_{t-1} / c)
        h_t | z_t     ~ Gamma(shape=nu + z_t, scale=c)

    Observation (Cox / Poisson with exposure and covariates):
        y_t | h_t, x_t ~ Poisson( h_t * tau_t * exp(x_t' beta) )

    Notes
    -----
    - Gamma uses NumPy parameterization: shape k, scale theta.
    - If you set X=0 and tau=1, you get y_t | h_t ~ Poisson(h_t),
      which is the simplest "paper-like" special case.
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

    # method to verify the shapes of each parameters to avoid errors 
    @staticmethod
    def _as_2d(X, T):
        """Ensure X is (T, p). Accepts None, (T,), (T,p)."""
        if X is None:
            return np.zeros((T, 0))
        X = np.asarray(X)
        if X.ndim == 1:
            if X.shape[0] != T:
                raise ValueError(f"X must have length T={T} if 1D.")
            return X.reshape(T, 1)
        if X.ndim == 2:
            if X.shape[0] != T:
                raise ValueError(f"X must have shape (T,p) with T={T}.")
            return X
        raise ValueError("X must be None, 1D (T,), or 2D (T,p).")

    @staticmethod
    def _as_1d_tau(tau, T):
        """Ensure tau is (T,), positive."""
        if tau is None:
            return np.ones(T)
        tau = np.asarray(tau, dtype=float)
        if tau.ndim != 1 or tau.shape[0] != T:
            raise ValueError(f"tau must be a 1D array of length T={T}.")
        if np.any(tau <= 0):
            raise ValueError("tau must be strictly positive.")
        return tau

    @staticmethod
    def _as_beta(beta, p):
        """Ensure beta is (p,). If p=0, return empty vector."""
        if p == 0:
            return np.zeros(0)
        if beta is None:
            raise ValueError("beta must be provided when X has at least one column.")
        beta = np.asarray(beta, dtype=float)
        if beta.ndim != 1 or beta.shape[0] != p:
            raise ValueError(f"beta must be a 1D array of length p={p}.")
        return beta

    def simulate(
        self,
        init="stationary",
        burn_in=500,
        h0=1.0,
        X=None,
        beta=None,
        tau=None,
        return_lambda=False,
        return_exposure=False,
    ):
        """
        Parameters
        ----------
        init : {"stationary", "burnin"}
            - "stationary": h0 ~ Gamma(nu, c/(1-phi)), simulate T steps
            - "burnin": start at h0, simulate burn_in + T, keep last T
        burn_in : int
            Burn-in length if init="burnin"
        h0 : float
            Initial value if init="burnin"
        X : array-like, shape (T,p) or (T,)
            Covariates x_t.
            If None -> no covariates (p=0).
        beta : array-like, shape (p,)
            Regression coefficients.
            Must be provided if X has p>0 columns.
        tau : array-like, shape (T,)
            Exposure (period length, number at risk, etc.). Must be positive.
            If None -> tau_t = 1.
        return_lambda : bool
            If True, also return lambda_t = phi * h_{t-1} / c (Poisson rate for z_t)
        return_exposure : bool
            If True, also return expo_t = tau_t * exp(x_t' beta)

        Returns
        -------
        y : ndarray (T,)
        h : ndarray (T,)
        z : ndarray (T,)
        (optional) lam_z : ndarray (T,)
        (optional) expo : ndarray (T,)
        """

        if init not in {"stationary", "burnin"}:
            raise ValueError("init must be 'stationary' or 'burnin'")

        # decide total length (burn-in + kept)
        if init == "stationary":
            burn = 0
            total = self.T
            h_prev = self.rng.gamma(shape=self.nu, scale=self.c / (1.0 - self.phi))
        else:
            burn = int(burn_in)
            total = self.T + burn
            h_prev = float(h0)

        # covariates/exposure for total horizon
        X_full = self._as_2d(X, self.T)
        tau_full = self._as_1d_tau(tau, self.T)

        # if burn-in, we need covariates/exposure of length total
        # simplest policy: repeat first row/value during burn-in
        if burn > 0:
            if X_full.shape[1] == 0:
                X_total = np.zeros((total, 0))
            else:
                X_total = np.vstack([np.repeat(X_full[:1, :], burn, axis=0), X_full])
            tau_total = np.concatenate([np.repeat(tau_full[:1], burn), tau_full])
        else:
            X_total = X_full
            tau_total = tau_full

        p = X_total.shape[1]
        beta_vec = self._as_beta(beta, p)

        # storage
        h = np.zeros(total)
        z = np.zeros(total, dtype=int)
        y = np.zeros(total, dtype=int)
        lam_z = np.zeros(total) if return_lambda else None
        expo = np.zeros(total) if return_exposure else None

        for t in range(total):
            # z_t | h_{t-1}
            lam_t = (self.phi * h_prev) / self.c
            if return_lambda:
                lam_z[t] = lam_t
            z[t] = self.rng.poisson(lam_t)

            # h_t | z_t
            h[t] = self.rng.gamma(shape=self.nu + z[t], scale=self.c)

            # exposure part: tau_t * exp(x_t' beta)
            if p == 0:
                expo_t = tau_total[t]
            else:
                expo_t = tau_total[t] * np.exp(X_total[t] @ beta_vec)
            if return_exposure:
                expo[t] = expo_t

            # y_t | h_t, x_t
            rate_y = h[t] * expo_t
            y[t] = self.rng.poisson(lam=rate_y)

            h_prev = h[t]

        # drop burn-in if needed
        if burn > 0:
            h = h[burn:]
            z = z[burn:]
            y = y[burn:]
            if return_lambda:
                lam_z = lam_z[burn:]
            if return_exposure:
                expo = expo[burn:]

        outs = [y, h, z]
        if return_lambda:
            outs.append(lam_z)
        if return_exposure:
            outs.append(expo)
        return tuple(outs)