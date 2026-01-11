import pandas as pd
import numpy as np
import ast

def load_data(filename, serie_name):
    """
    Loads Cox data from an Excel file.
    Handles Beta = None even if Pandas converts them to NaN.
    """
    try:
        # 1. Read metadata (Params et Beta)
        df_info = pd.read_excel(filename, sheet_name="INFO", index_col="Serie")
        
        # Retrieval
        params_str = df_info.loc[serie_name, "Params"]
        beta_raw = df_info.loc[serie_name, "Beta"]
        
        # Parameters parsing 
        params = ast.literal_eval(params_str)
        
        # We check whether it is a NaN (Pandas) OR the string ‘None’.
        if pd.isna(beta_raw) or str(beta_raw) == "None":
            beta = None
        else:
            # We force conversion to string before evaluating
            beta = np.array(ast.literal_eval(str(beta_raw)))
        # ----------------------

        # 2. Read raw data
        df = pd.read_excel(filename, sheet_name=serie_name)
        
        y = df["y"].values.astype(float)
        h_true = df["h_true"].values.astype(float)
        
        # 3. Rebuild X
        cov_cols = [c for c in df.columns if c.startswith("X_")]
        if cov_cols:
            X = df[cov_cols].values
        else:
            X = None
            
        print(f"✅ Load: {serie_name} (Beta: {beta})")
        return y, X, h_true, beta, params

    except FileNotFoundError:
        print(f"❌ Error: The file '{filename}' cannot be found.")
        return None, None, None, None, None
    except Exception as e:
        print(f"❌ Unexpected error : {e}")
        # We note the error to see the traceback if necessary
        raise e
