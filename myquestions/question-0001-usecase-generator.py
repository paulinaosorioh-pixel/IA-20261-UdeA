import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def generar_caso_de_uso_0001():

    n_rows = np.random.randint(800, 1200)
    ref_rows = np.random.randint(300, 500)
    n_sensors = np.random.randint(3, 7)
    sensor_names = [f"Sensor_{chr(65+i)}" for i in range(n_sensors)]


    data = np.random.normal(50, 5, (n_rows, n_sensors))
    df = pd.DataFrame(data, columns=sensor_names)


    n_drifting = np.random.randint(1, n_sensors)
    drifting_cols = np.random.choice(sensor_names, n_drifting, replace=False).tolist()

    expected_output = []

    for col in sensor_names:
        if col in drifting_cols:
            # Aplicamos un desplazamiento fuerte (ej. +4 desviaciones estándar)
            # a más del 15% de los datos finales para asegurar que falle
            start_drift = ref_rows + 1
            # El desplazamiento debe ser suficiente para salir de +/- 3 sigma
            df.loc[start_drift:, col] += 20
            expected_output.append(col)

    expected_output.sort()

    input_dict = {
        "df": df,
        "ref_rows": ref_rows
    }

    return input_dict, expected_output
