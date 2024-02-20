mlopsfinal
==============================

Estructura de repositorio para operaciones de Machine Learning

Organización del Proyecto
------------

    ├── .gitignore             <- Archivos y directorios a ignorar en git.
    ├── LICENSE
    ├── README.md              <- README de nivel superior para desarrolladores que usan este proyecto.
    ├── requirements.txt       <- El archivo de requerimientos para reproducir el entorno de análisis.
    ├── setup.py               <- Hace que el proyecto sea instalable con pip (pip install -e .) para importar src.
    ├── test_environment.py    <- Pruebas para el entorno de análisis.
    ├── tox.ini                <- Archivo tox con configuraciones para ejecutar tox.
    │
    ├── data
    │   ├── processed          <- Conjuntos de datos finales, canónicos, para modelado.
    │   └── raw                <- El volcado original de datos inmutables.
    │
    ├── notebooks              <- Notebooks de Jupyter. Convención de nombres: número (para ordenar),
    │                             iniciales del creador y una descripción corta delimitada por `-`.
    │
    ├── reports                <- Análisis generados como HTML, PDF, LaTeX, etc.
    │   └── figures            <- Gráficos y figuras generados para usar en informes.
    │
    └── src
        ├── __init__.py        <- Hace src un módulo de Python.
        ├── evaluate.py        <- Scripts para evaluar los modelos entrenados.
        ├── make_dataset.py    <- Scripts para descargar o generar datos.
        ├── predict.py         <- Scripts para hacer predicciones con modelos entrenados.
        └── train.py           <- Scripts para entrenar modelos.


--------
