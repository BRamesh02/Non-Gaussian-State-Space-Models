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
    Creal (2017) Cox / frailty model, for bootstrap PF on h_t, with time-varying exposure:

      h0 ~ Gamma(nu, scale=c/(1-phi))
      z_t | h_{t-1} ~ Poisson(phi*h_{t-1}/c)
      h_t | z_t ~ Gamma(nu+z_t, scale=c)
      y_t | h_t ~ Poisson(expo_t * h_t)

    Pass expo as a length-T vector (e.g. expo2 from your simulator).
    """
    def __init__(self, nu, phi, c, expo, seed=0, z_trunc_logpdf=300):
        self.nu = float(nu)
        self.phi = float(phi)
        self.c = float(c)
        self.expo = np.asarray(expo, dtype=float).ravel()
        self.rng = np.random.default_rng(seed)
        self.z_trunc_logpdf = int(z_trunc_logpdf)

    def PX0(self):
        # particles Gamma(a,b) uses b=rate, so scale = 1/b
        # want scale=c/(1-phi) -> rate=(1-phi)/c
        rate0 = (1.0 - self.phi) / self.c
        return dists.Gamma(a=self.nu, b=rate0)

    def PX(self, t, xp):
        # Return a transition object that can sample h_t given xp (vector)
        return GammaPoissonGammaTrans(
            hp=xp,
            nu=self.nu,
            phi=self.phi,
            c=self.c,
            rng=self.rng,
            z_trunc_logpdf=self.z_trunc_logpdf
        )

    def PY(self, t, xp, x):
        # y_t | h_t ~ Poisson(expo_t * h_t)
        return dists.Poisson(rate=self.expo[t] * x)
    

def run_particles_pf(y, expo, nu, phi, c, N=20000, seed=0, verbose=False):
    """
    Run bootstrap PF using particles, return total log-likelihood estimate.
    """
    fk = ssm.Bootstrap(
        ssm=CrealCoxSSM(nu=nu, phi=phi, c=c, expo=expo, seed=seed),
        data=y
    )
    alg = particles.SMC(fk=fk, N=N, verbose=verbose)
    alg.run()
    ll = alg.logLt
    return float(ll) if np.isscalar(ll) else float(np.sum(ll))