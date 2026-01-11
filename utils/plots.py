import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
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
    Plot MCMC diagnostics (traceplot, density, ACF).
    
    Arguments:
    ----------
    chains : np.ndarray
        Array of shape (n_chains, n_iter, n_params) or (n_iter, n_params).
        This is your 'res' object.
    true_params : list/array (optional)
        True parameter values for comparison (e.g., [0.8, 3, 0.4]).
    param_names : list (optional)
        Parameter names (e.g., [r'$\phi$', r'$\nu$', r'$c$']).
    burn_in : int
        Number of initial iterations to discard (warm-up).
    """
    
    # Handle dimensions (if a single chain is passed)
    if chains.ndim == 2:
        chains = chains[np.newaxis, :, :]
    
    n_chains, n_iter, n_params = chains.shape
    
    # Default names if not provided
    if param_names is None:
        param_names = [f'Param {i+1}' for i in range(n_params)]
        
    # Colors and style
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_chains))
    
    # Create figure: one row per parameter, three columns (Trace, Hist, ACF)
    fig, axes = plt.subplots(n_params, 3, figsize=(15, 4 * n_params))
    
    for i in range(n_params):
        # --- Data for this parameter ---
        # Keep only the post-burn-in samples
        chain_data = chains[:, burn_in:, i]
        flat_data = chain_data.flatten() # For the global histogram
        
        # 1. TRACEPLOT (Mélange)
        ax_trace = axes[i, 0]
        for c in range(n_chains):
            ax_trace.plot(chain_data[c], lw=0.5, alpha=0.6, color=colors[c])
        
        if true_params is not None:
            ax_trace.axhline(true_params[i], color='red', linestyle='--', lw=2, label='True')
            if i == 0: ax_trace.legend()
            
        ax_trace.set_title(f'Traceplot : {param_names[i]}')
        ax_trace.set_xlabel('Iterations (post burn-in)')
        ax_trace.set_ylabel('Value')
        ax_trace.grid(True, alpha=0.3)

        
        # 2. DENSITÉ (Posterior)
        ax_hist = axes[i, 1]
        # Histogram
        ax_hist.hist(flat_data, bins=30, density=True, alpha=0.4, color='skyblue', edgecolor='black')
        
        # KDE (smooth density estimate)
        kde = stats.gaussian_kde(flat_data)
        x_grid = np.linspace(flat_data.min(), flat_data.max(), 200)
        ax_hist.plot(x_grid, kde(x_grid), color='blue', lw=2)
        
        # True value line
        if true_params is not None:
            ax_hist.axvline(true_params[i], color='red', linestyle='--', lw=2, label='True')
            
        # Estimated mean
        estim_mean = np.mean(flat_data)
        ax_hist.axvline(estim_mean, color='green', linestyle='-', lw=2, label=f'Mean: {estim_mean:.3f}')
        
        ax_hist.set_title(f'Posterior Density : {param_names[i]}')
        if i == 0: ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)

        
        # 3. ACF (Autocorrélation)
        ax_acf = axes[i, 2]
        # Compute ACF on the first chain (usually sufficient for diagnostics)
        # Alternatively average ACFs across chains, but chain 0 is enough here.
        acf_vals = acf(chains[0, burn_in:, i], nlags=40)
        
        ax_acf.stem(range(len(acf_vals)), acf_vals, basefmt=" ")
        ax_acf.set_ylim(-0.2, 1.1)
        ax_acf.set_title(f'Autocorrelation (ACF) : {param_names[i]}')
        ax_acf.set_xlabel('Lag')
        ax_acf.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_pf_profile_grid(
    dfS,
    series_name,
    N_list,
    param_col="phi",
    loglik_col="loglik_pf",
    true_col=None,
    true_value=None,
    x_min=None,
    x_max=None,
    R_SHOW=5,
    Y_Q=(0.01, 0.99),
    RNG_SEED=123,
    xlabel=None,
    ncols=3,                # <<< 3 per row by default
    figsize_per_ax=(4.5, 3.8)
):
    """
    Grid plot of PF likelihood profiles over any parameter.
    - ncols fixed (default 3)
    - nrows adapts to len(N_list) (can be > 6)
    dfS must already be filtered to ONE series.
    """

    rng = np.random.default_rng(RNG_SEED)
    N_list = list(N_list)

    # resolve true value
    if true_value is None:
        if true_col is None:
            true_col = f"{param_col}_true"
        true_value = float(dfS[true_col].iloc[0]) if true_col in dfS.columns else None

    if xlabel is None:
        xlabel = rf"${param_col}$"

    n = len(N_list)
    nrows = int(np.ceil(n / ncols))
    fig_w = figsize_per_ax[0] * ncols
    fig_h = figsize_per_ax[1] * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharey=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, N in zip(axes, N_list):
        g = dfS[dfS["N"] == N].copy()

        # x truncation
        if x_min is not None:
            g = g[g[param_col] >= x_min]
        if x_max is not None:
            g = g[g[param_col] <= x_max]

        reps = np.sort(g["rep"].unique())
        reps_sel = rng.choice(reps, size=min(R_SHOW, len(reps)), replace=False)

        vals = []
        for r in reps_sel:
            gr = g[g["rep"] == r].sort_values(param_col)
            yv = gr[loglik_col].to_numpy()
            vals.append(yv)
            ax.plot(gr[param_col], yv, lw=1.0, alpha=0.7)

        # robust y-lims
        if vals:
            vals = np.concatenate(vals)
            y_lo, y_hi = np.quantile(vals, Y_Q)
            m = 0.05 * (y_hi - y_lo)
            ax.set_ylim(y_lo - m, y_hi + m)

        # true value line
        if true_value is not None:
            if (x_min is None or true_value >= x_min) and (x_max is None or true_value <= x_max):
                ax.axvline(true_value, color="red", ls="--", lw=1.2)

        # x-lims
        if len(g) > 0:
            xmin_plot = float(g[param_col].min()) if x_min is None else x_min
            xmax_plot = float(g[param_col].max()) if x_max is None else x_max
            ax.set_xlim(xmin_plot, xmax_plot)

        ax.set_title(f"N={N}")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)

    # turn off unused axes
    for ax in axes[len(N_list):]:
        ax.axis("off")

    axes[0].set_ylabel("log-likelihood estimate (PF)")

    # clean global legend (always neat)
    handles = [Line2D([0], [0], color="black", lw=1.5, alpha=0.7, label=f"PF runs (R_SHOW={R_SHOW})")]
    if true_value is not None:
        handles.append(Line2D([0], [0], color="red", lw=1.2, ls="--", label=f"True {param_col}"))

    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.995))

    fig.suptitle(f"PF likelihood profiles — {series_name}", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_mc_var_sd_and_hat_grid(
    df,
    series,
    param_col,
    true_col,
    loglik_col="loglik_pf",
    x_min=None,
    x_max=None,
    ncols=3,
    figsize_per_cell=(5.2, 3.6),
    box_ylim_quantiles=(0.01, 0.99),   
):
    """
    For each N: left = MC sd of log-likelihood vs parameter,
               right = boxplot of parameter-hat (argmax over grid) across reps.

    Required columns in df: ['series','N','rep', param_col, loglik_col, true_col]
    """

    dS = df[df["series"] == series].copy()
    if dS.empty:
        raise ValueError(f"Series '{series}' not found. Available: {sorted(df['series'].unique())}")

    for col in ["N", "rep", param_col, loglik_col, true_col]:
        if col not in dS.columns:
            raise ValueError(f"Missing column '{col}' in dataframe.")

    true_val = float(dS[true_col].iloc[0])
    N_LIST = sorted(dS["N"].unique())


    all_hat = []
    for N in N_LIST:
        gN = dS[dS["N"] == N].copy()
        if x_min is not None:
            gN = gN[gN[param_col] >= x_min]
        if x_max is not None:
            gN = gN[gN[param_col] <= x_max]
        if len(gN) == 0:
            continue
        idxmaxN = gN.groupby("rep")[loglik_col].idxmax()
        all_hat.append(gN.loc[idxmaxN, param_col].to_numpy())

    if all_hat:
        all_hat = np.concatenate(all_hat)
        qlo, qhi = box_ylim_quantiles
        y_lo, y_hi = np.quantile(all_hat, [qlo, qhi])
        margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.05
        box_ylim = (y_lo - margin, y_hi + margin)
    else:
        box_ylim = None


    # Figure grid
    n = len(N_LIST)
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows))
    outer = fig.add_gridspec(nrows, ncols, wspace=0.25, hspace=0.45)

    for idx, N in enumerate(N_LIST):
        r, c = divmod(idx, ncols)
        inner = outer[r, c].subgridspec(1, 2, wspace=0.30)
        ax_sd = fig.add_subplot(inner[0, 0])
        ax_box = fig.add_subplot(inner[0, 1])

        g = dS[dS["N"] == N].copy()

        # x truncation
        if x_min is not None:
            g = g[g[param_col] >= x_min]
        if x_max is not None:
            g = g[g[param_col] <= x_max]

        # left: sd across reps at each param value
        sd_by_param = (g.groupby(param_col)[loglik_col]
                         .std(ddof=1)
                         .reset_index(name="sd_mc")
                         .sort_values(param_col))

        ax_sd.plot(sd_by_param[param_col], sd_by_param["sd_mc"], lw=2)
        if (x_min is None or true_val >= x_min) and (x_max is None or true_val <= x_max):
            ax_sd.axvline(true_val, color="red", ls="--", lw=1.2)

        if len(sd_by_param) > 0:
            xmin_plot = float(sd_by_param[param_col].min()) if x_min is None else x_min
            xmax_plot = float(sd_by_param[param_col].max()) if x_max is None else x_max
            ax_sd.set_xlim(xmin_plot, xmax_plot)

        ax_sd.set_xlabel(rf"${param_col}$")
        ax_sd.set_ylabel("MC sd of log-likelihood")
        ax_sd.grid(alpha=0.25)
        ax_sd.set_title(f"N={N} (sd vs {param_col})")

        # right: hat(param) per rep = argmax over grid
        if len(g) > 0:
            idxmax = g.groupby("rep")[loglik_col].idxmax()
            param_hat = g.loc[idxmax, param_col].to_numpy()
        else:
            param_hat = np.array([])

        ax_box.boxplot([param_hat], labels=[f"N={N}"], showfliers=True)
        ax_box.axhline(true_val, color="red", ls="--", lw=1.2)


        if box_ylim is not None:
            ax_box.set_ylim(*box_ylim)

        ax_box.set_ylabel(rf"$\hat{{{param_col}}}$ (argmax)")
        ax_box.grid(alpha=0.25)
        ax_box.set_title(rf"$\hat{{{param_col}}}$ across reps")

    fig.suptitle(f"Monte Carlo variability — {series} — parameter: {param_col}", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_exact_vs_pf_cloud_by_rep_grid(
    df_pf, df_exact, series, param_col,
    pf_col="loglik_pf", exact_col="loglik_exact",
    true_col=None,
    x_min=None, x_max=None,
    N_list=None,
    ncols=3,
    figsize_per_ax=(4.8, 3.6),
    pf_alpha=0.5, pf_size=12,
    cmap_name="tab10",
    R_SHOW=1,
    seed=123,
):
    dP = df_pf[df_pf["series"] == series].copy()
    dE = df_exact[df_exact["series"] == series].copy()
    
    if true_col is None:
        true_col = f"{param_col}_true"
    true_val = float(dP[true_col].iloc[0]) if true_col in dP.columns else None

    if N_list is None:
        N_list = sorted(dP["N"].unique())
    else:
        N_list = list(N_list)

    if x_min is not None:
        dE = dE[dE[param_col] >= x_min]
    if x_max is not None:
        dE = dE[dE[param_col] <= x_max]
    dE = dE.sort_values(param_col)

    n = len(N_list)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows)
    )
    axes = np.atleast_1d(axes).ravel()

    rng = np.random.default_rng(seed)

    for ax, N in zip(axes, N_list):
        g = dP[dP["N"] == N].copy()
        if x_min is not None:
            g = g[g[param_col] >= x_min]
        if x_max is not None:
            g = g[g[param_col] <= x_max]

        reps = np.sort(g["rep"].unique())
        if len(reps) == 0:
            ax.set_title(f"{series} | N={N}")
            ax.axis("off")
            continue

        R_use = len(reps) if (R_SHOW is None) else min(int(R_SHOW), len(reps))
        reps_sel = reps if (R_SHOW is None or R_use == len(reps)) else rng.choice(reps, size=R_use, replace=False)

        cmap = cm.get_cmap(cmap_name, len(reps_sel))
        for j, r in enumerate(reps_sel):
            gr = g[g["rep"] == r]
            ax.scatter(gr[param_col], gr[pf_col], s=pf_size, alpha=pf_alpha, color=cmap(j))

        ax2 = ax.twinx()
        ax2.plot(dE[param_col], dE[exact_col], color="black", lw=2.2)

        if true_val is not None and (x_min is None or true_val >= x_min) and (x_max is None or true_val <= x_max):
            ax.axvline(true_val, color="red", ls="--", lw=1.2)

        xmin_plot = float(g[param_col].min()) if x_min is None else x_min
        xmax_plot = float(g[param_col].max()) if x_max is None else x_max
        ax.set_xlim(xmin_plot, xmax_plot)

        ax.set_title(f"{series} | N={N}")
        ax.set_xlabel(rf"${param_col}$")
        ax.set_ylabel("PF log-likelihood")
        ax2.set_ylabel("Exact log-likelihood")
        ax.grid(alpha=0.25)

    for ax in axes[len(N_list):]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], color="black", lw=2.2, label="Exact filter"),
        Line2D([0], [0], color="red", lw=1.2, ls="--", label="True parameter"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=7,
               markerfacecolor="gray", alpha=pf_alpha, label="PF path (one MC run)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))

    fig.suptitle(f"Exact vs PF likelihood — {series} — {param_col}", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
