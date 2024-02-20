import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics
from typing import Tuple
import matplotlib.pyplot as plt

def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Carga los datos de entrenamiento desde un archivo NPZ.

    Args:
        filename (str): Nombre del archivo NPZ que contiene los datos.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tupla con los datos de entrada (X) y las etiquetas (y).
    """
    data_path = os.path.join('../data/processed', filename)
    train_data = np.load(data_path)
    X_train, y_train = train_data['X'], train_data['y']
    return X_train, y_train


def build_model(hidden_dim1: int, hidden_dim2: int, dropout_rate: float = 0.5) -> models.Sequential:
    """Construye y retorna un modelo secuencial de Keras.

    Args:
        hidden_dim1 (int): Número de filtros para la primera capa convolucional.
        hidden_dim2 (int): Número de filtros para la segunda capa convolucional.
        dropout_rate (float, optional): Tasa de dropout para las capas de dropout. Por defecto es 0.5.

    Returns:
        models.Sequential: Modelo secuencial construido.
    """
    model = models.Sequential([
        layers.Conv2D(filters=hidden_dim1, kernel_size=(5, 5), padding='same', activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Conv2D(filters=hidden_dim2, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Conv2D(filters=hidden_dim2, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Flatten(),
        layers.Dense(10, activation="softmax")
    ])

    model.build(input_shape=(None, 28, 28, 1))

    return model


def plot_and_save_accuracy(history, output_dir='../reports/figures', file_name='training_accuracy.png'):
    """Genera y guarda un gráfico de la precisión del entrenamiento y validación.

    Args:
        history (tf.keras.callbacks.History): Historia del entrenamiento del modelo.
        output_dir (str, optional): Directorio de salida para guardar el gráfico. Por defecto 'reports/figures'.
        file_name (str, optional): Nombre del archivo para guardar el gráfico. Por defecto 'training_accuracy.png'.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
    plt.title('Precisión durante el Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend(loc='upper left')
    plt.xlim(0, 2)

    # Guardar la figura
    plt.savefig(os.path.join(output_dir, file_name))
    print(f'Gráfico guardado en {os.path.join(output_dir, file_name)}')



def train_model(model: models.Sequential, X_train: np.ndarray, y_train: np.ndarray,
                learning_rate: float, epochs: int, batch_size: int) -> tf.keras.callbacks.History:
    """Configura y entrena el modelo.

    Args:
        model (models.Sequential): Modelo de Keras a entrenar.
        X_train (np.ndarray): Datos de entrada de entrenamiento.
        y_train (np.ndarray): Etiquetas de entrenamiento.
        learning_rate (float): Tasa de aprendizaje para el optimizador.
        epochs (int): Número de épocas de entrenamiento.
        batch_size (int): Tamaño del lote para el entrenamiento.

    Returns:
        tf.keras.callbacks.History: Historia del entrenamiento del modelo.
    """
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss=losses.SparseCategoricalCrossentropy(),
                  metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    plot_and_save_accuracy(history, output_dir='../reports/figures', file_name='training_accuracy.png')

    # Guardar el modelo
    model_save_path = 'models/best_model.h5'
    if not os.path.isdir('models'):
        os.makedirs('models')
    model.save(model_save_path)
    print(f'Modelo guardado en {model_save_path}')

    print('El entrenamiento del modelo ha finalizado con éxito.')

    return history


def main() -> None:
    """Función principal para el entrenamiento del modelo."""
    # Hiperparámetros
    LR = 1e-3
    EPOCHS = 2
    BATCH_SIZE = 64
    CONV_DIM1 = 56
    CONV_DIM2 = 100

    X_train, y_train = load_data('train_data.npz')
    model = build_model(CONV_DIM1, CONV_DIM2)
    history = train_model(model, X_train, y_train, LR, EPOCHS, BATCH_SIZE)

    print('El entrenamiento del modelo ha finalizado con éxito.')


if __name__ == "__main__":
    main()
