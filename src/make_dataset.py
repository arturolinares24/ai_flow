import os
import zipfile
from typing import Tuple

import pandas as pd
import numpy as np
import wget
from sklearn.model_selection import train_test_split


def download_data(data_path: str) -> None:
    """Descarga los datos necesarios para el procesamiento desde una URL y los guarda en una carpeta local.

    Args:
        data_path (str): Ruta de la carpeta donde se guardarán los datos descargados.
    """
    train_link = 'https://github.com/kubeflow/examples/blob/master/digit-recognition-kaggle-competition/data/train.csv.zip?raw=true'
    wget.download(train_link, os.path.join(data_path, 'data_csv.zip'))

    with zipfile.ZipFile(os.path.join(data_path, 'data_csv.zip'), "r") as zip_ref:
        zip_ref.extractall(data_path)


def read_file_csv(filename: str, data_path: str) -> pd.DataFrame:
    """Lee un archivo CSV y devuelve un DataFrame.

    Args:
        filename (str): Nombre del archivo CSV.
        data_path (str): Ruta de la carpeta donde se encuentra el archivo CSV.

    Returns:
        pd.DataFrame: DataFrame creado a partir del archivo CSV.
    """
    train_data_path = os.path.join(data_path, filename)
    df = pd.read_csv(train_data_path)
    print(filename, 'cargado correctamente')
    return df


def data_process(dataframe: pd.DataFrame) -> None:
    """Procesa los datos, dividiéndolos en conjuntos de entrenamiento, prueba y validación, y los guarda en archivos NPZ.

    Args:
        dataframe (pd.DataFrame): DataFrame que contiene los datos a procesar.
    """
    ntrain = dataframe.shape[0]

    all_data_X = dataframe.drop('label', axis=1)
    all_data_y = dataframe.label

    all_data_X = all_data_X.values.reshape(-1, 28, 28, 1)
    all_data_X = all_data_X / 255.0

    X = all_data_X[:ntrain].copy()
    y = all_data_y[:ntrain].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=42)

    np.savez(os.path.join('../data/processed/', 'train_data.npz'), X=X_train, y=y_train)
    print('Data de entrenamiento exportada correctamente en la carpeta processed')

    np.savez(os.path.join('../data/processed/', 'test_data.npz'), X=X_test, y=y_test)
    print('Data de prueba exportada correctamente en la carpeta processed')

    np.savez(os.path.join('../data/processed/', 'val_data.npz'), X=X_val, y=y_val)
    print('Data de validacion exportada correctamente en la carpeta processed')


def main() -> None:
    """Función principal para la preparación de datos."""
    data_path = "../data/raw"
    download_data(data_path)
    df1 = read_file_csv('train.csv', data_path)
    data_process(df1)


if __name__ == "__main__":
    main()
