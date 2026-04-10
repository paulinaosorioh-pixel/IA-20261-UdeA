import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def generar_caso_de_uso_0003():

    n_features = np.random.randint(5, 15)
    n_train = np.random.randint(500, 1000)
    n_prod = np.random.randint(100, 200)

    cols = [f"feature_{i}" for i in range(n_features)]


    train_data = np.random.randn(n_train, n_features)
    train_df = pd.DataFrame(train_data, columns=cols)


    prod_data = np.random.randn(n_prod, n_features)
    n_anomalies = int(n_prod * 0.1)

    prod_data[:n_anomalies] = (prod_data[:n_anomalies] * 5) + 10


    for df in [train_df, pd.DataFrame(prod_data, columns=cols)]:
        for _ in range(5):
            df.iloc[np.random.randint(0, len(df)), np.random.randint(0, n_features)] = np.nan

    production_df = pd.DataFrame(prod_data, columns=cols)


    input_dict = {
        "train_df": train_df,
        "production_df": production_df
    }


    train_clean = train_df.interpolate(method='linear').bfill().ffill()
    prod_clean = production_df.interpolate(method='linear').bfill().ffill()


    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_clean)
    prod_scaled = scaler.transform(prod_clean)


    pca = PCA(n_components=0.95)
    pca.fit(train_scaled)


    train_reconstructed = pca.inverse_transform(pca.transform(train_scaled))
    train_errors = np.linalg.norm(train_scaled - train_reconstructed, axis=1)


    threshold = np.percentile(train_errors, 99)


    prod_reconstructed = pca.inverse_transform(pca.transform(prod_scaled))
    prod_errors = np.linalg.norm(prod_scaled - prod_reconstructed, axis=1)


    expected_output = production_df[prod_errors > threshold]

    return input_dict, expected_output
