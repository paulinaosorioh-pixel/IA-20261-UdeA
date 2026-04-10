import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def generar_caso_de_uso_0002():

    n_samples = np.random.randint(100, 500)
    n_features = np.random.randint(2, 6)
    threshold = round(np.random.uniform(0.1, 0.5), 2)

    col_names = [f"feature_{i}" for i in range(n_features)]

    data_train = np.random.randn(n_samples, n_features)
    df_train = pd.DataFrame(data_train, columns=col_names)

    data_test = data_train.copy()
    expected_output = {}


    for i, col in enumerate(col_names):
        if np.random.choice([True, False]):
            shift = np.random.uniform(threshold * 2, threshold * 5)
            data_test[:, i] += shift

            expected_output[col] = float(np.abs(shift))

    df_test = pd.DataFrame(data_test, columns=col_names)


    input_dict = {
        "df_train": df_train,
        "df_test": df_test,
        "threshold": threshold
    }

    output_dict = {k: v for k, v in expected_output.items() if v > threshold}

    return input_dict, output_dict

