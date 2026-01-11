import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy import stats
from statsmodels.tsa.stattools import acf


# ---- 1) Time series y and h ----
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


# ---- 2) Histogram of h and theoretical density ----
def plot_histogram_h(h, nu, phi, c, burn=100):
    h = np.asarray(h).ravel()
    h_ss = h[burn:] if burn < len(h) else h

    x = np.linspace(0, np.percentile(h_ss, 99.5), 300)
    pdf = gamma.pdf(x, a=nu, scale=c/(1-phi))

    # averages (small stationarity check)
    emp_mean = float(np.mean(h_ss))
    theo_mean = nu * (c/(1-phi))

    plt.figure(figsize=(6, 4))
    plt.hist(h_ss, bins=40, density=True, alpha=0.5)
    plt.plot(x, pdf, linewidth=2)

    plt.axvline(emp_mean, linestyle="--", linewidth=2, label=f"Emp. mean={emp_mean:.3g}")
    plt.axvline(theo_mean, linestyle=":", linewidth=2, label=f"Theo. mean={theo_mean:.3g}")

    plt.title("Histogram of h_t + stationary Gamma density")
    plt.xlabel("h")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---- 3) Autocorrelation of h ----
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


# ---- 3bis) Autocorrelation of y (useful because it is observed) ----
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

    # h_t : red line (latent)
    ax1.plot(
        t,
        h[start:end],
        color="red",
        linewidth=2,
        label=r"Latent intensity $h_t$"
    )
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$h_t$")

    # y_t : black line (observed)
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

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.show()




def plot_mcmc_diagnostics(chains, true_params=None, param_names=None, burn_in=500):
    """
    Trace un diagnostic complet pour un algorithme MCMC (Traceplot, Densité, ACF).
    
    Arguments:
    ----------
    chains : np.ndarray
        Tableau de forme (n_chains, n_iter, n_params) ou (n_iter, n_params).
        C'est votre objet 'res'.
    true_params : list/array (optionnel)
        Les vraies valeurs des paramètres pour comparaison (ex: [0.8, 3, 0.4]).
    param_names : list (optionnel)
        Noms des paramètres (ex: [r'$\phi$', r'$\nu$', r'$c$']).
    burn_in : int
        Nombre d'itérations à ignorer au début (période de chauffe).
    """
    
    # Gestion des dimensions (si une seule chaîne est passée)
    if chains.ndim == 2:
        chains = chains[np.newaxis, :, :]
    
    n_chains, n_iter, n_params = chains.shape
    
    # Noms par défaut si non fournis
    if param_names is None:
        param_names = [f'Param {i+1}' for i in range(n_params)]
        
    # Couleurs et style
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_chains))
    
    # Création de la figure : Une ligne par paramètre, 3 colonnes (Trace, Hist, ACF)
    fig, axes = plt.subplots(n_params, 3, figsize=(15, 4 * n_params))
    
    for i in range(n_params):
        # --- Données pour ce paramètre ---
        # On ne garde que la partie post-burn-in
        chain_data = chains[:, burn_in:, i]
        flat_data = chain_data.flatten() # Pour l'histogramme global
        
        # -----------------------------
        # 1. TRACEPLOT (Mélange)
        # -----------------------------
        ax_trace = axes[i, 0]
        for c in range(n_chains):
            ax_trace.plot(chain_data[c], lw=0.5, alpha=0.6, color=colors[c])
        
        if true_params is not None:
            ax_trace.axhline(true_params[i], color='red', linestyle='--', lw=2, label='Vrai')
            if i == 0: ax_trace.legend()
            
        ax_trace.set_title(f'Traceplot : {param_names[i]}')
        ax_trace.set_xlabel('Itérations (post burn-in)')
        ax_trace.set_ylabel('Valeur')
        ax_trace.grid(True, alpha=0.3)

        # -----------------------------
        # 2. DENSITÉ (Posterior)
        # -----------------------------
        ax_hist = axes[i, 1]
        # Histogramme
        ax_hist.hist(flat_data, bins=30, density=True, alpha=0.4, color='skyblue', edgecolor='black')
        
        # KDE (Estimation de densité lisse)
        kde = stats.gaussian_kde(flat_data)
        x_grid = np.linspace(flat_data.min(), flat_data.max(), 200)
        ax_hist.plot(x_grid, kde(x_grid), color='blue', lw=2)
        
        # Ligne de la vraie valeur
        if true_params is not None:
            ax_hist.axvline(true_params[i], color='red', linestyle='--', lw=2, label='Vrai')
            
        # Moyenne estimée
        estim_mean = np.mean(flat_data)
        ax_hist.axvline(estim_mean, color='green', linestyle='-', lw=2, label=f'Moy: {estim_mean:.3f}')
        
        ax_hist.set_title(f'Densité A Posteriori : {param_names[i]}')
        if i == 0: ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)

        # -----------------------------
        # 3. ACF (Autocorrélation)
        # -----------------------------
        ax_acf = axes[i, 2]
        # On calcule l'ACF sur la première chaîne (souvent suffisant pour le diagnostic)
        # ou sur la moyenne des ACF si on veut être puriste, mais ici chaîne 0 suffit.
        acf_vals = acf(chains[0, burn_in:, i], nlags=40)
        
        ax_acf.stem(range(len(acf_vals)), acf_vals, basefmt=" ")
        ax_acf.set_ylim(-0.2, 1.1)
        ax_acf.set_title(f'Autocorrélation (ACF) : {param_names[i]}')
        ax_acf.set_xlabel('Lag')
        ax_acf.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

