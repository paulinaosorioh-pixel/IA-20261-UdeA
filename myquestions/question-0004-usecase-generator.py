
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generar_caso_de_uso_0004():
    n_rows = np.random.randint(500, 1000)
    noise_level = np.random.uniform(0.1, 0.5)
    num_anomalies = np.random.randint(5, 15)

    fechas = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_rows)]

    t = np.linspace(0, 4 * np.pi * (n_rows/24), n_rows)
    consumo = 50 + 20 * np.sin(t) + np.random.normal(0, noise_level, n_rows)

    indices_anomalias = np.random.choice(range(n_rows), size=num_anomalies, replace=False)
    consumo[indices_anomalias] *= np.random.uniform(2.0, 5.0)

    df_input = pd.DataFrame({
        'timestamp': fechas,
        'consumo_kwh': consumo
    })

    input_dict = {"df_consumo": df_input}

    df_output_mock = df_input.copy()
    df_output_mock['es_anomalia'] = False

    expected_ratio_placeholder = float(num_anomalies / (n_rows * 0.2))

    output_tuple = (df_output_mock, expected_ratio_placeholder)

    return input_dict, output_tuple
