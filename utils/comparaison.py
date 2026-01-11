import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import particles
import particles.state_space_models as ssm
from src.particles_library import CrealCoxSSM
import src.creal_filter as cf



def pf_run(y, nu, phi, c, X=None, beta=None, tau=None, N=1000, seed=0):
    y = np.asarray(y, dtype=np.int64)
    T = len(y)

    fk = ssm.Bootstrap(
        ssm=CrealCoxSSM(
            nu=float(nu), phi=float(phi), c=float(c),
            T=T, X=X, beta=beta, tau=tau, expo=None, seed=int(seed)
        ),
        data=y
    )
    alg = particles.SMC(fk=fk, N=int(N), verbose=False, store_history=True)
    alg.run()

    X_hist = alg.hist.X
    W_hist = np.array([alg.hist.wgts[t].W for t in range(T)])

    ll = alg.logLt
    ll = float(ll) if np.isscalar(ll) else float(np.sum(ll))
    return ll, X_hist, W_hist


def exact_mean(y, nu, c, X, beta, tau, filt_prob):
    y = np.asarray(y, dtype=np.int64)
    T = len(y)

    tau = np.ones(T) if tau is None else np.asarray(tau, float).ravel()
    if X is None:
        expo = tau
    else:
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(T, 1)
        beta = np.asarray(beta, float).ravel()
        expo = tau * np.exp(X @ beta)

    z = np.arange(filt_prob.shape[1], dtype=float)
    Eh = np.empty(T, dtype=float)

    for t in range(T):
        scale = c / (1.0 + c * float(expo[t]))
        shape = nu + float(y[t]) + z
        Eh[t] = np.sum(filt_prob[t] * (shape * scale))

    return Eh


def run_exact_pf(y, X, beta, tau, nu, phi, c, Z=500, N=2000, seed=123):
    y = np.asarray(y, dtype=np.int64)

    f = cf.ExactFilter(y, Z_trunc=int(Z))
    ll_exact, filt_prob = f.log_likelihood(
        phi=float(phi), nu=float(nu), c=float(c),
        X=X, coeffs=beta, tau=tau,
        return_filter=True
    )
    Eh_exact = exact_mean(y, float(nu), float(c), X, beta, tau, filt_prob)

    ll_pf, X_hist, W_hist = pf_run(y, nu, phi, c, X=X, beta=beta, tau=tau, N=N, seed=seed)
    Eh_pf = np.array([np.sum(W_hist[t] * X_hist[t]) for t in range(len(y))], dtype=float)

    return ll_exact, ll_pf, Eh_exact, Eh_pf


def plot_three(Eh_exact, Eh_pf, h_true=None, T_max=None, name=""):
    T = len(Eh_exact)
    tmax = T if T_max is None else min(int(T_max), T)
    sl = slice(0, tmax)

    ht = None if h_true is None else np.asarray(h_true, float)

    # 1) True vs Exact
    plt.figure(figsize=(10, 4))
    if ht is not None:
        plt.plot(ht[sl], alpha=0.35, label="True $h_t$")
    plt.plot(Eh_exact[sl], lw=2, label=r"Exact $E[h_t\mid y_{1:t}]$")
    plt.title((name + " — " if name else "") + "True vs Exact")
    plt.xlabel("t"); plt.ylabel(r"$h_t$")
    plt.legend(); plt.tight_layout(); plt.show()

    # 2) True vs PF (ligne)
    plt.figure(figsize=(10, 4))
    if ht is not None:
        plt.plot(ht[sl], alpha=0.35, label="True $h_t$")
    plt.plot(Eh_pf[sl], lw=2, label=r"PF $E[h_t\mid y_{1:t}]$")
    plt.title((name + " — " if name else "") + "True vs PF")
    plt.xlabel("t"); plt.ylabel(r"$h_t$")
    plt.legend(); plt.tight_layout(); plt.show()

    # 3) True vs Exact vs PF (PF en points)
    plt.figure(figsize=(10, 4))
    if ht is not None:
        plt.plot(ht[sl], alpha=0.35, label="True $h_t$")
    plt.plot(Eh_exact[sl], lw=2, label=r"Exact $E[h_t\mid y_{1:t}]$")
    plt.plot(Eh_pf[sl], lw=0, marker="o", markersize=2.5, alpha=0.75,
             label=r"PF $E[h_t\mid y_{1:t}]$ (points)")
    plt.title((name + " — " if name else "") + "True vs Exact vs PF")
    plt.xlabel("t"); plt.ylabel(r"$h_t$")
    plt.legend(); plt.tight_layout(); plt.show()


def compare_numeric(Eh_exact, Eh_pf, h_true=None, T_max=None):
    Eh_exact = np.asarray(Eh_exact, float)
    Eh_pf = np.asarray(Eh_pf, float)

    T = len(Eh_exact)
    tmax = T if T_max is None else min(int(T_max), T)
    sl = slice(0, tmax)

    d = Eh_pf[sl] - Eh_exact[sl]

    out = {
        "T_used": int(tmax),
        "RMSE_pf_vs_exact": float(np.sqrt(np.mean(d ** 2))),
        "MAE_pf_vs_exact": float(np.mean(np.abs(d))),
        "Bias_pf_vs_exact": float(np.mean(d)),
        "MaxAbs_pf_vs_exact": float(np.max(np.abs(d))),
    }

    if h_true is not None:
        ht = np.asarray(h_true, float)
        out["RMSE_exact_vs_true"] = float(np.sqrt(np.mean((Eh_exact[sl] - ht[sl]) ** 2)))
        out["RMSE_pf_vs_true"] = float(np.sqrt(np.mean((Eh_pf[sl] - ht[sl]) ** 2)))

    return out


def metrics_to_df(metrics, extra=None):
    d = dict(metrics)
    if extra:
        d.update(extra)
    return pd.DataFrame([d])