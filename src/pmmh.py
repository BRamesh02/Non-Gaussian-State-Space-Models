import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

import particles
from particles import state_space_models as ssm
from particles import distributions as dists
from particles.mcmc import PMMH
from particles_library import CrealCoxSSM

class CoxAnalysisManager:
    def __init__(self, n_iter=2000, burnin=500):
        self.n_iter = n_iter
        self.burnin = burnin
        self.results = {}

    def _make_prior(self, has_beta=False, dim_beta=0):
        """Dynamically builds the prior dictionary."""
        p = {
            'phi': dists.Uniform(0.0, 0.99),
            'nu':  dists.Gamma(2.0, 1.0),
            'c':   dists.Gamma(2.0, 1.0)
        }
        if has_beta:
            p['beta'] = dists.MvNormal(loc=np.zeros(dim_beta), cov=np.eye(dim_beta))
        return dists.StructDist(p)

    def _get_init_theta(self, prior, has_beta=False, dim_beta=0):
        """Initializes theta in a stable region."""
        theta = prior.rvs(size=1)
        theta['phi'] = 0.6
        theta['nu'] = 2.0
        theta['c'] = 1.0
        if has_beta:
            theta['beta'] = np.zeros(dim_beta)
        return theta

    def run_analysis(self, name, y_data, X_data=None, tau_data=None):
        """Runs the standard visual analysis (Trace, Density, ACF)."""
        print(f"\n{'='*10} Starting Analysis: {name} {'='*10}")
        
        has_beta = (X_data is not None)
        dim_beta = 0
        if has_beta:
            X_data = np.asarray(X_data)
            if X_data.ndim == 1: X_data = X_data.reshape(-1, 1)
            dim_beta = X_data.shape[1]
            print(f"-> Mode: Regression with {dim_beta} covariate(s).")
        else:
            print("-> Mode: No covariates (fixed tau/expo).")

        prior = self._make_prior(has_beta, dim_beta)
        theta0 = self._get_init_theta(prior, has_beta, dim_beta)

        # Dynamic Wrapper class to inject specific data (X, tau, y)
        class LocalWrapper(CrealCoxSSM):
            def __init__(self, **kwargs):
                super().__init__(X=X_data, tau=tau_data, T=len(y_data), **kwargs)

        chains = {}
        for n_part, label in [(200, 'Low N'), (1500, 'High N')]:
            print(f"-> Launching PMMH {label} ({n_part} particles)...")
            pmmh = PMMH(
                niter=self.n_iter,
                ssm_cls=LocalWrapper,
                data=y_data,
                prior=prior,
                Nx=n_part,
                theta0=theta0,
                verbose=False # Silence verbose output for cleaner logs
            )
            pmmh.run()
            chains[label] = pmmh.chain

        self.results[name] = chains
        self._plot_results(name, chains, has_beta)

    def _plot_results(self, name, chains, has_beta):
        """
        Plotting:
        1. Trace (Time series of the chain)
        2. Density (KDE)
        3. ACF (Autocorrelation) - Now includes BOTH Low N and High N
        """
        chain_low = chains['Low N']
        chain_high = chains['High N']
        
        params = ['phi', 'nu', 'c']
        if has_beta:
            params.append('beta')

        n_plots = len(params)
        # 3 Columns: Trace | Density | ACF
        fig, axes = plt.subplots(n_plots, 3, figsize=(16, 3.5 * n_plots))
        fig.suptitle(f"Full Results: {name}", fontsize=16)

        # Handle 1D axes array if only one parameter
        if n_plots == 1: axes = np.array([axes])

        for i, param in enumerate(params):
            # Extract data based on parameter type
            if param == 'beta':
                d_low = chain_low.theta['beta'][:, 0]
                d_high = chain_high.theta['beta'][:, 0]
                lbl = "Beta[0]"
            else:
                d_low = chain_low.theta[param]
                d_high = chain_high.theta[param]
                lbl = param

            # Apply burn-in
            # Using safe slicing to avoid empty array errors if burnin >= n_iter
            safe_burn = min(self.burnin, len(d_low) - 1)
            data_low = d_low[safe_burn:]
            data_high = d_high[safe_burn:]

            # --- 1. Trace Plot ---
            ax_trace = axes[i, 0]
            ax_trace.plot(d_high, label='High N', alpha=0.6)
            ax_trace.plot(d_low, label='Low N', alpha=0.6, color='red', lw=1)
            ax_trace.set_title(f"Trace - {lbl}")
            ax_trace.legend()

            # --- 2. Density Plot ---
            ax_kde = axes[i, 1]
            try:
                if len(data_high) > 1:
                    sns.kdeplot(data_high, ax=ax_kde, label='High N', fill=True)
                    sns.kdeplot(data_low, ax=ax_kde, label='Low N', color='red', fill=True)
                    ax_kde.set_title(f"Density - {lbl}")
                else:
                    ax_kde.text(0.5, 0.5, "Not enough data", ha='center')
            except:
                ax_kde.text(0.5, 0.5, "KDE Error", ha='center')

            # --- 3. ACF Plot (Dual) ---
            ax_acf = axes[i, 2]
            n_samples = len(data_high)
            
            # Dynamic lag calculation to prevent "ValueError" on short runs
            safe_lags = min(40, n_samples // 2 - 1)
            if safe_lags < 1: safe_lags = 1

            if n_samples > 2:
                # A. Plot Low N first (Background, Red, NO Confidence Interval to avoid clutter)
                plot_acf(data_low, ax=ax_acf, lags=safe_lags, alpha=None, 
                         color='red', vlines_kwargs={"colors": "red", "alpha": 0.3},
                         title=f"ACF - {lbl}")
                
                # B. Plot High N on top (Foreground, Blue, WITH Confidence Interval)
                # We overwrite the title to clean up the double plotting artifacts
                plot_acf(data_high, ax=ax_acf, lags=safe_lags, alpha=0.05, 
                         color='tab:blue', vlines_kwargs={"colors": "tab:blue"},
                         title=f"ACF (Red=LowN, Blue=HighN) - {lbl}")
            else:
                ax_acf.text(0.5, 0.5, "Not enough data", ha='center')

        plt.tight_layout()
        plt.show()

    def run_stability_test(self, name, y_data, X_data=None, tau_data=None, n_repeats=10, Nx=300):
        """
        Runs the algorithm multiple times (Monte Carlo of Monte Carlo) 
        to evaluate the variance/stability of the results.
        """
        print(f"\n{'='*10} Stability Test (Variance): {name} {'='*10}")
        print(f"-> Launching {n_repeats} independent runs with N={Nx} particles...")

        has_beta = (X_data is not None)
        dim_beta = 0
        if has_beta:
            X_data = np.asarray(X_data)
            if X_data.ndim == 1: X_data = X_data.reshape(-1, 1)
            dim_beta = X_data.shape[1]

        prior = self._make_prior(has_beta, dim_beta)

        class LocalWrapper(CrealCoxSSM):
            def __init__(self, **kwargs):
                super().__init__(X=X_data, tau=tau_data, T=len(y_data), **kwargs)

        repeated_means = []
        
        for i in range(n_repeats):
            # Randomize start slightly
            theta0 = prior.rvs(size=1) 
            theta0['phi'] = np.random.uniform(0.4, 0.8) 
            
            pmmh = PMMH(
                niter=self.n_iter,
                ssm_cls=LocalWrapper,
                data=y_data,
                prior=prior,
                Nx=Nx,
                theta0=theta0,
                verbose=False 
            )
            pmmh.run()
            
            # Safe burn-in slicing
            safe_burn = min(self.burnin, self.n_iter - 1)
            
            stats = {}
            stats['phi'] = np.mean(pmmh.chain.theta['phi'][safe_burn:])
            stats['nu'] = np.mean(pmmh.chain.theta['nu'][safe_burn:])
            stats['c'] = np.mean(pmmh.chain.theta['c'][safe_burn:])
            
            if has_beta:
                betas = pmmh.chain.theta['beta'][safe_burn:] 
                mean_betas = np.mean(betas, axis=0)
                for k in range(dim_beta):
                    stats[f'beta_{k}'] = mean_betas[k]
            
            repeated_means.append(stats)
            print(f"   Run {i+1}/{n_repeats} completed.")

        df_res = pd.DataFrame(repeated_means)
        self._plot_stability(name, df_res)
        return df_res

    def _plot_stability(self, name, df):
        """Displays boxplots of the estimated posterior means."""
        cols = df.columns
        n_vars = len(cols)
        
        fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 5))
        if n_vars == 1: axes = [axes]
        
        fig.suptitle(f"Estimation Variability ({len(df)} runs) : {name}", fontsize=16)

        for i, col in enumerate(cols):
            ax = axes[i]
            
            # --- SEABORN FIX: use data=df and y=col name ---
            sns.boxplot(data=df, y=col, ax=ax, color='skyblue')
            sns.swarmplot(data=df, y=col, ax=ax, color='black', alpha=0.7)
            
            mean_val = df[col].mean()
            std_val = df[col].std()
            cv_pct = (std_val / abs(mean_val)) * 100 if mean_val != 0 else 0
            
            ax.set_title(f"{col}\nMean: {mean_val:.3f} | CV: {cv_pct:.1f}%")
            ax.set_ylabel("Estimated Posterior Mean")

        plt.tight_layout()
        plt.show()
