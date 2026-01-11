import math
import numpy as np
from numba import njit

@njit
def _logsumexp_1d(a):
    """Stable log-sum-exp written explicitly for Numba."""
    m = -1e300  # -inf
    for i in range(a.shape[0]):
        if a[i] > m:
            m = a[i]
            
    if m == -1e300:
        return -1e300

    s = 0.0
    for i in range(a.shape[0]):
        s += math.exp(a[i] - m)

    return m + math.log(s)


@njit
def _nbinom_logpmf(k, r, p):
    """Log-PMF of the Negative Binomial distribution."""
    if k < 0: return -1e300
    return (math.lgamma(k + r)
            - math.lgamma(r)
            - math.lgamma(k + 1.0)
            + r * math.log(p)
            + k * math.log(1.0 - p))


@njit
def _exact_filter_core(y, exposure_arr, Z, phi, nu, c, return_filter):
    T = y.shape[0]
    z_grid = np.arange(Z + 1)

    # Init: z_1 ~ NB(r=nu, p=1-phi)
    log_p_z = np.empty(Z + 1)
    p_init = 1.0 - phi
    for i in range(Z + 1):
        log_p_z[i] = _nbinom_logpmf(z_grid[i], nu, p_init)

    norm = _logsumexp_1d(log_p_z)
    for i in range(Z + 1):
        log_p_z[i] -= norm

    total_log_like = 0.0
    max_pZ = 0.0

    tmp = np.empty(Z + 1)
    new_log_p_z = np.empty(Z + 1)

    # Store filtered probabilities if requested
    if return_filter:
        filt_prob = np.empty((T, Z + 1))
    else:
        filt_prob = np.empty((1, 1))  # dummy

    for t in range(T):
        yt = y[t]
        bt = exposure_arr[t]

        # Update
        p_obs = 1.0 / (1.0 + c * bt)
        for i in range(Z + 1):
            r_obs = nu + z_grid[i]
            tmp[i] = _nbinom_logpmf(yt, r_obs, p_obs) + log_p_z[i]

        log_like_t = _logsumexp_1d(tmp)
        total_log_like += log_like_t

        for i in range(Z + 1):
            log_p_z[i] = tmp[i] - log_like_t

        # Filtered: p(z_t | y_1:t)
        if return_filter:
            for i in range(Z + 1):
                filt_prob[t, i] = math.exp(log_p_z[i])

        # Diagnostic: boundary probability
        pZ = math.exp(log_p_z[Z])
        if pZ > max_pZ:
            max_pZ = pZ

        # Prediction 
        if t < T - 1:
            p_trans = (1.0 + c * bt) / (1.0 + c * bt + phi)

            for j in range(Z + 1):
                for i in range(Z + 1):
                    r_trans = nu + yt + z_grid[i]
                    tmp[i] = _nbinom_logpmf(z_grid[j], r_trans, p_trans) + log_p_z[i]
                new_log_p_z[j] = _logsumexp_1d(tmp)

            norm_pred = _logsumexp_1d(new_log_p_z)
            for j in range(Z + 1):
                log_p_z[j] = new_log_p_z[j] - norm_pred

    return total_log_like, max_pZ, filt_prob


class ExactFilter:
    """Exact filter from Creal (2017) and Creal (2012), "A Class of Non-Gaussian State Space Models with Exact Likelihood Inference"."""
    def __init__(self, y, Z_trunc=50):
        self.y = np.asarray(y, dtype=np.int64)
        self.T = len(self.y)
        self.Z = int(Z_trunc)

    def _compute_exposure(self, expo, X, beta_coeffs, tau):
        # Direct exposure provided
        if expo is not None:
            return np.asarray(expo, dtype=np.float64).ravel()

        # Otherwise build from X, beta, tau
        if tau is None:
            tau_vec = np.ones(self.T)
        else:
            tau_vec = np.asarray(tau, dtype=np.float64).ravel()

        if X is None:
        # No regressors, just tau
            return tau_vec

        if beta_coeffs is None:
            raise ValueError("coeffs must be provided when X is provided")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(self.T, 1)

        beta_coeffs = np.asarray(beta_coeffs, dtype=np.float64).ravel()

        lin_pred = X @ beta_coeffs
        return tau_vec * np.exp(lin_pred)


    def log_likelihood(self, phi, nu, c,
                   X=None, coeffs=None, tau=None, exposure=None,
                   return_diag=False, return_filter=False):
        """
        Compute the log-likelihood.
        
        Parameters:
        -----------
        phi, nu, c : scalar model parameters
        X : (T, K) regressor matrix (optional)
        coeffs : (K,) regression coefficients (optional)
        tau : (T,) offset/volume (optional)
        exposure : (T,) direct exposure vector (takes priority if provided)
        """
        # Prepare data (Python/Numpy)
        exposure_vec = self._compute_exposure(exposure, X, coeffs, tau)
        
        if len(exposure_vec) != self.T:
            raise ValueError(f"Exposure must have length T={self.T}")
            
        # Call optimized core (Numba)
        # The JIT compiler caches this function on first call
        ll, max_pz, filt_prob = _exact_filter_core(
            self.y, exposure_vec, self.Z, float(phi), float(nu), float(c),
            return_filter
        )

        if return_filter and return_diag:
            return ll, max_pz, filt_prob
        if return_filter:
            return ll, filt_prob
        if return_diag:
            return ll, max_pz
        return ll
