import math
import numpy as np
from numba import njit


# ============================================================
# Numba helpers
# ============================================================

@njit
def _logsumexp_1d(a):
    """
    Stable log-sum-exp for a 1D array.
    """
    m = -1e300
    for i in range(a.shape[0]):
        if a[i] > m:
            m = a[i]

    s = 0.0
    for i in range(a.shape[0]):
        s += math.exp(a[i] - m)

    return m + math.log(s)


@njit
def _nbinom_logpmf(k, r, p):
    """
    Negative Binomial log-pmf (SciPy parameterization):

        P(K = k) = C(k+r-1, k) (1-p)^k p^r

    where k = number of failures before r successes.
    """
    return (math.lgamma(k + r)
            - math.lgamma(r)
            - math.lgamma(k + 1.0)
            + r * math.log(p)
            + k * math.log(1.0 - p))


@njit
def _exact_filter_loglik_beta_vec(y, beta, Z, phi, nu, c):
    """
    Exact truncated filter with time-varying beta_t.

    Parameters
    ----------
    y : array[int], shape (T,)
    beta : array[float], shape (T,)
        beta_t >= 0
    Z : int
    phi, nu, c : float

    Returns
    -------
    total_log_like : float
    max_pZ : float
    """
    T = y.shape[0]
    z_grid = np.arange(Z + 1)

    # --------------------------------------------------------
    # Initialization: z_1 ~ NB(r=nu, p=1-phi)
    # (kept as in your code for internal consistency)
    # --------------------------------------------------------
    log_p_z = np.empty(Z + 1)
    p_init = 1.0 - phi
    for i in range(Z + 1):
        log_p_z[i] = _nbinom_logpmf(z_grid[i], nu, p_init)

    total_log_like = 0.0
    max_pZ = 0.0

    tmp = np.empty(Z + 1)
    new_log_p_z = np.empty(Z + 1)

    # --------------------------------------------------------
    # Filtering recursion
    # --------------------------------------------------------
    for t in range(T):
        yt = y[t]
        bt = beta[t]  # beta_t

        # Observation: y_t | z_t  ~ NB(r=nu+z_t, p_obs)
        # with p_obs = 1/(1 + c*beta_t)   (SciPy NB parameterization)
        p_obs = 1.0 / (1.0 + c * bt)

        # ---- Update step
        for i in range(Z + 1):
            r_obs = nu + z_grid[i]
            tmp[i] = _nbinom_logpmf(yt, r_obs, p_obs) + log_p_z[i]

        log_like_t = _logsumexp_1d(tmp)
        total_log_like += log_like_t

        for i in range(Z + 1):
            log_p_z[i] = tmp[i] - log_like_t

        # Diagnostic: mass at truncation boundary
        pZ = math.exp(log_p_z[Z])
        if pZ > max_pZ:
            max_pZ = pZ

        # ---- Prediction step
        if t < T - 1:
            y_prev = yt
            b_prev = bt

            # Transition: z_{t+1} | z_t, y_t ~ NB(r=nu+z_t+y_t, p_trans)
            # with p_trans = (1 + c*beta_t)/(1 + c*beta_t + phi)
            p_trans = (1.0 + c * b_prev) / (1.0 + c * b_prev + phi)

            for j in range(Z + 1):
                for i in range(Z + 1):
                    r_trans = nu + y_prev + z_grid[i]
                    tmp[i] = _nbinom_logpmf(z_grid[j], r_trans, p_trans) + log_p_z[i]
                new_log_p_z[j] = _logsumexp_1d(tmp)

            for j in range(Z + 1):
                log_p_z[j] = new_log_p_z[j]

    return total_log_like, max_pZ


@njit
def _exact_filter_loglik_beta_scalar(y, beta, Z, phi, nu, c):
    """
    Same as above but beta is a scalar (faster).
    """
    T = y.shape[0]
    z_grid = np.arange(Z + 1)

    # init: z_1 ~ NB(r=nu, p=1-phi)
    log_p_z = np.empty(Z + 1)
    p_init = 1.0 - phi
    for i in range(Z + 1):
        log_p_z[i] = _nbinom_logpmf(z_grid[i], nu, p_init)

    # precompute obs/trans p's (constant beta)
    p_obs = 1.0 / (1.0 + c * beta)
    p_trans = (1.0 + c * beta) / (1.0 + c * beta + phi)

    total_log_like = 0.0
    max_pZ = 0.0

    tmp = np.empty(Z + 1)
    new_log_p_z = np.empty(Z + 1)

    for t in range(T):
        yt = y[t]

        # update
        for i in range(Z + 1):
            r_obs = nu + z_grid[i]
            tmp[i] = _nbinom_logpmf(yt, r_obs, p_obs) + log_p_z[i]

        log_like_t = _logsumexp_1d(tmp)
        total_log_like += log_like_t

        for i in range(Z + 1):
            log_p_z[i] = tmp[i] - log_like_t

        pZ = math.exp(log_p_z[Z])
        if pZ > max_pZ:
            max_pZ = pZ

        # prediction
        if t < T - 1:
            y_prev = yt
            for j in range(Z + 1):
                for i in range(Z + 1):
                    r_trans = nu + y_prev + z_grid[i]
                    tmp[i] = _nbinom_logpmf(z_grid[j], r_trans, p_trans) + log_p_z[i]
                new_log_p_z[j] = _logsumexp_1d(tmp)

            for j in range(Z + 1):
                log_p_z[j] = new_log_p_z[j]

    return total_log_like, max_pZ


# ============================================================
# User-facing class
# ============================================================

class ExactFilter:
    """
    Exact (truncated) likelihood filter for the Poisson–Gamma
    count model (Creal 2017, model used in Sec. 5.1 in aggregated form).

    Only approximation: truncation z_t ∈ {0,...,Z_trunc}.

    Parameters
    ----------
    y : array-like
        Observed count time series.
    Z_trunc : int
        Truncation level for the latent state z_t.

    Usage
    -----
    >>> f = ExactFilter(y, Z_trunc=200)
    >>> ll, max_pZ = f.log_likelihood(phi, nu, c, beta=30.0, return_diag=True)
    """

    def __init__(self, y, Z_trunc=50):
        self.y = np.asarray(y, dtype=np.int64)
        if self.y.ndim != 1:
            raise ValueError("y must be a 1D array of counts.")
        if np.any(self.y < 0):
            raise ValueError("y must contain nonnegative counts.")

        self.T = self.y.shape[0]
        self.Z = int(Z_trunc)
        if self.Z < 0:
            raise ValueError("Z_trunc must be >= 0.")

    def log_likelihood(self, phi, nu, c, beta=1.0, return_diag=False):
        """
        Compute log-likelihood via the exact truncated filter.

        beta can be:
          - a positive scalar (constant beta)
          - an array-like of length T (time-varying beta_t)
        """
        phi = float(phi)
        nu = float(nu)
        c = float(c)

        if not (0.0 < phi < 1.0):
            raise ValueError("phi must satisfy 0 < phi < 1.")
        if not (nu > 1.0):
            raise ValueError("nu must be > 1 (Feller condition).")
        if not (c > 0.0):
            raise ValueError("c must be > 0.")

        # beta handling
        if np.isscalar(beta):
            beta = float(beta)
            if not (beta > 0.0):
                raise ValueError("beta must be > 0.")
            ll, max_pZ = _exact_filter_loglik_beta_scalar(self.y, beta, self.Z, phi, nu, c)
        else:
            beta = np.asarray(beta, dtype=np.float64)
            if beta.ndim != 1 or beta.shape[0] != self.T:
                raise ValueError("beta must be a scalar or a 1D array of length T.")
            if np.any(beta <= 0.0):
                raise ValueError("beta_t must be > 0 for all t.")
            ll, max_pZ = _exact_filter_loglik_beta_vec(self.y, beta, self.Z, phi, nu, c)

        if return_diag:
            return ll, max_pZ
        return ll