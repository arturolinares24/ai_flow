import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import os
from tensorflow.keras.models import load_model
from typing import Tuple


def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Carga los datos de entrenamiento desde un archivo NPZ.

    Args:
        filename (str): Nombre del archivo NPZ que contiene los datos.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tupla con los datos de entrada (X) y las etiquetas (y).
    """
    data_path = os.path.join("../data/processed", filename)
    train_data = np.load(data_path)
    X_train, y_train = train_data["X"], train_data["y"]
    return X_train, y_train


def eval_model(X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Evalúa el modelo cargado con los datos de prueba y muestra la matriz de confusión.

    Args:
        X_test (np.ndarray): Datos de entrada de prueba.
        y_test (np.ndarray): Etiquetas de prueba.

    Returns:
        None
    """
    # Ruta al modelo guardado
    model_path = "models/best_model.h5"

    # Cargar el modelo
    model = load_model(model_path)

    print("Modelo cargado exitosamente")

    # Evaluación del modelo
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test_loss: {test_loss}, Test_accuracy: {test_acc}")

    # Predicciones del modelo
    y_pred = np.argmax(model.predict(X_test), axis=-1)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Visualización de la matriz de confusión
    plt.figure(figsize=(7, 7))
    sns.heatmap(cm, fmt="g", cbar=False, annot=True, cmap="Blues")
    plt.title("Matriz de confusión")
    plt.ylabel("Etiqueta real")
    plt.xlabel("Etiqueta predicha")
    # Guardar la figura
    output_dir = '../reports/figures'
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_test.png'))




def main() -> None:
    """Función principal para la evaluación del modelo."""
    X_test, y_test = load_data("test_data.npz")

    eval_model(X_test, y_test)
    print("Finalizó la evaluación del modelo")


if __name__ == "__main__":
    main()
