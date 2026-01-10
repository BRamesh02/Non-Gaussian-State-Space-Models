import pandas as pd
import numpy as np
import ast

def load_data(filename, serie_name):
    """
    Charge les données Cox depuis un fichier Excel.
    Gère les Beta = None même si Pandas les transforme en NaN.
    """
    try:
        # 1. Lire les métadonnées (Params et Beta)
        df_info = pd.read_excel(filename, sheet_name="INFO", index_col="Serie")
        
        # Récupération
        params_str = df_info.loc[serie_name, "Params"]
        beta_raw = df_info.loc[serie_name, "Beta"]
        
        # Parsing des paramètres (dictionnaire)
        params = ast.literal_eval(params_str)
        
        # --- CORRECTION ICI ---
        # On vérifie si c'est un NaN (Pandas) OU la string "None"
        if pd.isna(beta_raw) or str(beta_raw) == "None":
            beta = None
        else:
            # On force la conversion en string avant d'évaluer
            beta = np.array(ast.literal_eval(str(beta_raw)))
        # ----------------------

        # 2. Lire les données brutes
        df = pd.read_excel(filename, sheet_name=serie_name)
        
        y = df["y"].values.astype(float)
        h_true = df["h_true"].values.astype(float)
        
        # 3. Reconstruire X
        cov_cols = [c for c in df.columns if c.startswith("X_")]
        if cov_cols:
            X = df[cov_cols].values
        else:
            X = None
            
        print(f"✅ Chargé: {serie_name} (Beta: {beta})")
        return y, X, h_true, beta, params

    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier '{filename}' est introuvable.")
        return None, None, None, None, None
    except Exception as e:
        print(f"❌ Erreur inattendue : {e}")
        # On relève l'erreur pour voir le traceback si besoin
        raise e