import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from statsmodels.tsa.stattools import acf


# ---- 1) Série temporelle y et h ----
def plot_time_series(y, h, T_show=200, start=0, use_bars=False):
    y = np.asarray(y).ravel()
    h = np.asarray(h).ravel()
    T = min(len(y), len(h))
    end = min(T, start + T_show)

    t = np.arange(start, end)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(t, h[start:end], linewidth=2)
    ax1.set_xlabel("t")
    ax1.set_ylabel("h_t")

    ax2 = ax1.twinx()
    if use_bars:
        ax2.bar(t, y[start:end], alpha=0.25, width=1.0)
    else:
        ax2.plot(t, y[start:end], ".", alpha=0.25)

    ax2.set_ylabel("y_t")
    plt.title("Série simulée : h_t (ligne) et y_t (points/barres)")
    plt.tight_layout()
    plt.show()


# ---- 2) Histogramme de h et densité théorique ----
def plot_histogram_h(h, nu, phi, c, burn=100):
    h = np.asarray(h).ravel()
    h_ss = h[burn:] if burn < len(h) else h

    x = np.linspace(0, np.percentile(h_ss, 99.5), 300)
    pdf = gamma.pdf(x, a=nu, scale=c/(1-phi))

    # moyennes (petit check stationnarité)
    emp_mean = float(np.mean(h_ss))
    theo_mean = nu * (c/(1-phi))

    plt.figure(figsize=(6, 4))
    plt.hist(h_ss, bins=40, density=True, alpha=0.5)
    plt.plot(x, pdf, linewidth=2)

    plt.axvline(emp_mean, linestyle="--", linewidth=2, label=f"Emp. mean={emp_mean:.3g}")
    plt.axvline(theo_mean, linestyle=":", linewidth=2, label=f"Theo. mean={theo_mean:.3g}")

    plt.title("Histogramme de h_t + densité Gamma stationnaire")
    plt.xlabel("h")
    plt.ylabel("densité")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---- 3) Autocorrélation de h ----
def plot_acf_h(h, lags=30, burn=0):
    h = np.asarray(h).ravel()
    h2 = h[burn:] if burn < len(h) else h

    vals = acf(h2, nlags=lags, fft=True)
    plt.figure(figsize=(6, 4))
    plt.stem(range(lags+1), vals, basefmt=" ")
    plt.title("ACF de h_t")
    plt.xlabel("lag")
    plt.ylabel("ACF")
    plt.tight_layout()
    plt.show()


# ---- 3bis) Autocorrélation de y (utile car c'est observé) ----
def plot_acf_y(y, lags=30, burn=0):
    y = np.asarray(y).ravel()
    y2 = y[burn:] if burn < len(y) else y

    vals = acf(y2, nlags=lags, fft=True)
    plt.figure(figsize=(6, 4))
    plt.stem(range(lags+1), vals, basefmt=" ")
    plt.title("ACF de y_t")
    plt.xlabel("lag")
    plt.ylabel("ACF")
    plt.tight_layout()
    plt.show()
    

def plot_overlay_clean(y, h, T_show=400, start=0):
    y = np.asarray(y).ravel()
    h = np.asarray(h).ravel()
    T = min(len(y), len(h))
    end = min(T, start + T_show)
    t = np.arange(start, end)

    fig, ax1 = plt.subplots(figsize=(12, 4))

    # h_t : ligne rouge (latent)
    ax1.plot(
        t,
        h[start:end],
        color="red",
        linewidth=2,
        label=r"Latent intensity $h_t$"
    )
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$h_t$")

    # y_t : ligne noire (observé)
    ax2 = ax1.twinx()
    ax2.plot(
        t,
        y[start:end],
        color="black",
        linewidth=1.2,
        alpha=0.7,
        label=r"Observed counts $y_t$"
    )
    ax2.set_ylabel(r"$y_t$")

    plt.title(f"Simulation : Cox Process (T={len(y)})")

    # légende combinée
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.show()
