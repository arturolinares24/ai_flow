import os
from typing import Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Carga los datos de entrenamiento desde un archivo NPZ.

    Args:
        filename (str): Nombre del archivo NPZ.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Datos de entrada y etiquetas cargados.
    """
    data_path = os.path.join('../data/processed', filename)
    train_data = np.load(data_path)
    X_train, y_train = train_data['X'], train_data['y']
    return X_train, y_train


def score_model(X_val: np.ndarray, y_val: np.ndarray) -> None:
    """Evalúa el modelo cargado y guarda los resultados en un archivo CSV.

    Args:
        X_val (np.ndarray): Datos de entrada de validación.
        y_val (np.ndarray): Etiquetas de validación.
    """
    # Ruta al modelo guardado
    model_path = 'models/best_model.h5'

    # Cargar el modelo
    model = load_model(model_path)
    print("Modelo cargado exitosamente")

    y_pred = np.argmax(model.predict(X_val), axis=-1)

    results_df = pd.DataFrame({
        'Real': y_val,
        'Predicted': y_pred
    })
    results_df.to_csv(os.path.join('../data/scores/score.csv'),index=False)
    print('Exportación correctamente en la carpeta scores')


def main() -> None:
    """Función principal para la evaluación del modelo."""
    X_test, y_test = load_data('val_data.npz')
    score_model(X_test, y_test)
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()
