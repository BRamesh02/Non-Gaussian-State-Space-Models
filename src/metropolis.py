import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

# --- 1. FONCTIONS DE BASE ---

def log_prior(theta):
    """
    Définit les contraintes du modèle.
    theta = [phi, nu, c]
    """
    phi, nu, c = theta
    if 0 < phi < 0.999 and nu > 0 and c > 0:
        return 0.0
    else:
        return -np.inf

def run_metropolis_exact(y_data, exact_filter, n_iterations=5000, initial_theta=None, proposal_std=None, disable_tqdm=False):
    """
    Algorithme MCMC Random Walk Metropolis-Hastings.
    MODIFICATION : Retourne maintenant (chain, acceptance_rate)
    """
    if initial_theta is None:
        current_theta = np.array([0.5, 2.0, 0.5])
    else:
        current_theta = np.array(initial_theta)
        
    if proposal_std is None:
        proposal_std = np.array([0.02, 0.2, 0.1]) 
    
    chain = np.zeros((n_iterations, 3))
    accept_count = 0
    
    if log_prior(current_theta) == -np.inf:
        current_theta = np.array([0.5, 2.5, 0.5])
        
    current_log_prior = log_prior(current_theta)
    current_log_lik = exact_filter.log_likelihood(current_theta[0], current_theta[1], current_theta[2])
    current_log_post = current_log_lik + current_log_prior

    iterator = range(n_iterations)
    if not disable_tqdm:
        iterator = tqdm(iterator, desc="MCMC Sampling")
    
    for i in iterator:
        proposal = current_theta + np.random.normal(0, proposal_std)
        prop_log_prior = log_prior(proposal)
        
        if prop_log_prior == -np.inf:
            chain[i] = current_theta
        else:
            try:
                prop_log_lik = exact_filter.log_likelihood(proposal[0], proposal[1], proposal[2])
                prop_log_post = prop_log_lik + prop_log_prior
                
                log_alpha = prop_log_post - current_log_post
                
                if np.log(np.random.rand()) < log_alpha:
                    current_theta = proposal
                    current_log_post = prop_log_post
                    accept_count += 1
            except Exception:
                pass
        
        chain[i] = current_theta

    # CALCUL DU TAUX
    acc_rate = accept_count / n_iterations
    
    # On affiche seulement si on n'est pas en mode "silencieux" (parallèle)
    if not disable_tqdm:
        print(f"Taux d'acceptation final : {acc_rate:.2%}")
        
    # RETOURNE UN TUPLE MAINTENANT
    return chain, acc_rate

# --- 2. FONCTION PARALLÈLE (LE WORKER) ---

def _worker_chain(seed, y, exact_filter, n_iter, proposal_std):
    np.random.seed(seed)
    start_phi = np.random.uniform(0.5, 0.95) # Attention à 0.999 c'est risqué
    start_nu  = np.random.uniform(1.5, 3.5)
    start_c   = np.random.uniform(0.1, 0.3) # Eviter 0 pile
    start_theta = [start_phi, start_nu, start_c]
    
    # Le worker renvoie (chain, rate)
    return run_metropolis_exact(
        y_data=y,
        exact_filter=exact_filter,
        n_iterations=n_iter,
        initial_theta=start_theta,
        proposal_std=proposal_std,
        disable_tqdm=True 
    )

# --- 3. ORCHESTRATEUR MULTI-CHAÎNES ---

def run_multi_chain_mcmc(y, exact_filter, n_chains=4, n_iter=2000, proposal_std=[0.008, 0.07, 0.03], burn_in=500, true_params=None):
    """
    Lance plusieurs chaînes MCMC en parallèle et affiche TOUS les résultats.
    """
    print(f"🚀 Lancement de {n_chains} chaînes MCMC en parallèle sur CPU...")
    
    # --- 1. EXÉCUTION PARALLÈLE ---
    results = Parallel(n_jobs=-1)(
        delayed(_worker_chain)(
            seed=k, 
            y=y, 
            exact_filter=exact_filter, 
            n_iter=n_iter, 
            proposal_std=proposal_std
        ) for k in tqdm(range(n_chains), desc="Progression globale")
    )
    
    # Décomposition des résultats
    chains = np.array([r[0] for r in results]) # Shape: (n_chains, n_iter, 3)
    rates = [r[1] for r in results]
    
    # --- AFFICHAGE COMPLET DES TAUX D'ACCEPTATION ---
    print("\n" + "="*40)
    print("      DÉTAIL DES TAUX D'ACCEPTATION")
    print("="*40)
    print(f"Moyenne globale : {np.mean(rates):.2%}\n")
    
    # On boucle sur TOUTES les chaînes, sans limite
    for k, r in enumerate(rates):
        # Indicateur visuel : Idéalement entre 20% et 50%
        if 0.20 <= r <= 0.50:
            status = "✅ Optimal"
        elif r < 0.10:
            status = "⚠️ Trop faible (Pas bloqué)"
        elif r > 0.60:
            status = "⚠️ Trop élevé (Marche aléatoire lente)"
        else:
            status = "👌 Acceptable"
            
        # k+1:02d permet d'aligner les chiffres (01, 02, ... 20)
        print(f"Chaîne {k+1:02d} : {r:.2%}  -> {status}")

    print("="*40)

    # --- 2. FIGURE 1 : TRACEPLOTS ---
    print("\n✅ Génération Figure 1 : Traceplots...")
    param_names = [r'$\phi$', r'$\nu$', r'$c$']
    colors = plt.cm.jet(np.linspace(0, 1, n_chains))
    
    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    for i in range(3):
        ax = axes1[i]
        for k in range(n_chains):
            ax.plot(chains[k][:, i], alpha=0.5, color=colors[k], lw=1)
        
        if true_params is not None:
            ax.axhline(true_params[i], color='black', linestyle='--', linewidth=2, label='Vrai')
            
        ax.set_ylabel(param_names[i])
        ax.set_title(f"Traceplot : {param_names[i]}")
        ax.grid(True, alpha=0.3)
        if i == 0 and true_params is not None: ax.legend(loc='upper right')
        
    plt.xlabel("Itérations")
    plt.tight_layout()
    plt.show()

    # --- 3. CALCUL DES MOYENNES ---
    chain_means = []
    for k in range(n_chains):
        clean_samples = chains[k][burn_in:]
        chain_means.append(clean_samples.mean(axis=0))
    chain_means = np.array(chain_means)

    # --- 4. FIGURE 2 : HISTOGRAMMES ---
    print("\n✅ Génération Figure 2 : Histogrammes des Estimateurs...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    for i in range(3):
        ax = axes2[i]
        vals = chain_means[:, i]
        n_bins = max(5, int(n_chains / 2))
        ax.hist(vals, bins=n_bins, color='skyblue', edgecolor='black', alpha=0.7)
        
        global_avg = np.mean(vals)
        ax.axvline(global_avg, color='red', linestyle='-', linewidth=2, label=f'Moyenne: {global_avg:.3f}')
        
        if true_params is not None:
            ax.axvline(true_params[i], color='green', linestyle='--', linewidth=2, label=f'Vrai: {true_params[i]}')
            
        ax.set_title(f"Distribution : {param_names[i]}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()
    
    # --- 5. STATISTIQUES FINALES ---
    all_samples = np.vstack([c[burn_in:] for c in chains])
    global_mean = all_samples.mean(axis=0)
    global_std = all_samples.std(axis=0)
    
    print("\n--- RÉSULTATS FINAUX (Agrégés) ---")
    print(f"Phi : {global_mean[0]:.4f} +/- {global_std[0]:.4f}")
    print(f"Nu  : {global_mean[1]:.4f} +/- {global_std[1]:.4f}")
    print(f"c   : {global_mean[2]:.4f} +/- {global_std[2]:.4f}")
    
    return chains