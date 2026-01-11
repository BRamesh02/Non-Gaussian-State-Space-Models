import math
import numpy as np

import particles
import particles.state_space_models as ssm
import particles.distributions as dists


def _logsumexp(a):
    a = np.asarray(a, dtype=float)
    m = np.max(a)
    if np.isneginf(m):
        return -np.inf
    return m + np.log(np.sum(np.exp(a - m)))


def _poisson_logpmf(k, lam):
    # k integer, lam > 0
    if lam <= 0:
        return -np.inf
    return k * np.log(lam) - lam - math.lgamma(k + 1.0)


def _gamma_logpdf(x, shape, scale):
    # Gamma(shape, scale), x>0
    if x <= 0 or shape <= 0 or scale <= 0:
        return -np.inf
    return (shape - 1.0) * np.log(x) - x / scale - math.lgamma(shape) - shape * np.log(scale)


class GammaPoissonGammaTrans:
    """
    Transition law for h_t | h_{t-1}=hp:
      z ~ Poisson(phi * hp / c)
      h ~ Gamma(shape=nu+z, scale=c)

    Provides:
      - rvs(size=...) used by Bootstrap filter
      - logpdf(x) optional (truncated mixture) for compatibility with some routines
    """
    def __init__(self, hp, nu, phi, c, rng, z_trunc_logpdf=300):
        self.hp = np.asarray(hp, dtype=float)  # vector of size N
        self.nu = float(nu)
        self.phi = float(phi)
        self.c = float(c)
        self.rng = rng
        self.z_trunc_logpdf = int(z_trunc_logpdf)

    def rvs(self, size=None):
        # In particles Bootstrap, size is typically N; we can ignore it and use hp shape
        lam = (self.phi * self.hp) / self.c
        z = self.rng.poisson(lam=lam)  # vector
        return self.rng.gamma(shape=self.nu + z, scale=self.c)  # vector

    def logpdf(self, x):
        """
        Truncated mixture density:
          p(h|hp) = sum_{z>=0} Poisson(z; phi*hp/c) * Gamma(h; nu+z, scale=c)

        We truncate z at z_trunc_logpdf for numerical evaluation.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x[None]
        if x.shape[0] != self.hp.shape[0]:
            # particles may call logpdf with x shaped (N,) matching hp
            # if not, we try to broadcast if x is scalar
            if x.size == 1:
                x = np.full_like(self.hp, float(x))
            else:
                raise ValueError("logpdf: x must have same length as hp (number of particles).")

        out = np.empty_like(self.hp, dtype=float)
        for i in range(self.hp.shape[0]):
            lam = (self.phi * self.hp[i]) / self.c
            # log-sum over z=0..Z
            logs = []
            for z in range(self.z_trunc_logpdf + 1):
                logs.append(
                    _poisson_logpmf(z, lam) + _gamma_logpdf(x[i], self.nu + z, self.c)
                )
            out[i] = _logsumexp(np.array(logs))
        return out


class CrealCoxSSM(ssm.StateSpaceModel):
    """
    Modèle:
      h0 ~ Gamma(nu, scale=c/(1-phi))
      z_t | h_{t-1} ~ Poisson(phi*h_{t-1}/c)
      h_t | z_t ~ Gamma(nu+z_t, scale=c)
      y_t | h_t ~ Poisson(expo_t * h_t)

    expo_t est défini par priorité:
      1) expo (si fourni)
      2) tau_t * exp(X_t @ beta) (si X et beta fournis)
      3) tau_t (si tau fourni)
      4) 1 (sinon)
    """
    def __init__(self, nu, phi, c, T=None,
                 expo=None, X=None, beta=None, tau=None,
                 seed=0, z_trunc_logpdf=300):

        self.nu = float(nu)
        self.phi = float(phi)
        self.c  = float(c)

        self.rng = np.random.default_rng(int(seed))
        self.z_trunc_logpdf = int(z_trunc_logpdf)

        # ---- déterminer T ----
        if expo is not None:
            expo_arr = np.asarray(expo, dtype=float).ravel()
            T_infer = expo_arr.shape[0]
        elif X is not None:
            X_arr = np.asarray(X, dtype=float)
            T_infer = X_arr.shape[0]
        elif tau is not None:
            tau_arr = np.asarray(tau, dtype=float).ravel()
            T_infer = tau_arr.shape[0]
        elif T is not None:
            T_infer = int(T)
        else:
            raise ValueError("Need T (or expo, or X, or tau) to infer time length.")

        self.T = T_infer

        # ---- construire tau ----
        if tau is None:
            self.tau = np.ones(self.T, dtype=float)
        else:
            self.tau = np.asarray(tau, dtype=float).ravel()
            if self.tau.shape[0] != self.T:
                raise ValueError("tau must have length T")

        # ---- cas 1: expo direct ----
        if expo is not None:
            self.expo = np.asarray(expo, dtype=float).ravel()
            if self.expo.shape[0] != self.T:
                raise ValueError("expo must have length T")
            self._mode = "expo"
            return

        # ---- cas 2: X beta ----
        if (X is not None) or (beta is not None):
            if X is None or beta is None:
                raise ValueError("If using regression, must provide BOTH X and beta.")
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.shape[0] != self.T:
                raise ValueError("X must have T rows")
            self.X = X

            self.beta = np.asarray(beta, dtype=float).ravel()
            if self.X.shape[1] != self.beta.shape[0]:
                raise ValueError(f"beta must have length K={self.X.shape[1]}")
            self._mode = "xbeta"
            return

        # ---- cas 3: pas de covariables, juste tau ----
        self._mode = "tau_only"

    def PX0(self):
        rate0 = (1.0 - self.phi) / self.c
        return dists.Gamma(a=self.nu, b=rate0)

    def PX(self, t, xp):
        return GammaPoissonGammaTrans(
            hp=xp, nu=self.nu, phi=self.phi, c=self.c,
            rng=self.rng, z_trunc_logpdf=self.z_trunc_logpdf
        )

    def PY(self, t, xp, x):
        if self._mode == "expo":
            expo_t = self.expo[t]
        elif self._mode == "xbeta":
            expo_t = self.tau[t] * math.exp(float(self.X[t] @ self.beta))
        else:  # "tau_only"
            expo_t = self.tau[t]
        return dists.Poisson(rate=expo_t * x)


