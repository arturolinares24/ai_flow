from make_dataset import main as make_dataset_main
from train import main as train_main
from evaluate import main as evaluate_main
from predict import main as predict_main

def main_flow():
    # Crear y procesar el conjunto de datos
    make_dataset_main()

    # Entrenar el modelo
    train_main()

    # Evaluar el modelo
    evaluate_main()

    # Realizar predicciones con el modelo
    predict_main()

if __name__ == "__main__":
    main_flow()