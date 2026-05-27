# BistroTech — Pipeline de Machine Learning Nativo (Local y S3)

Esta carpeta contiene la implementación nativa y autónoma (sin dependencias acopladas de infraestructura compleja en AWS como SageMaker Pipelines, Kinesis o EventBridge) del sistema de recomendación gastronómica de BistroTech.

## Características Principales
* **Ejecución Local Autónoma:** Permite simular datos y entrenar los modelos localmente de forma rápida y sencilla.
* **Reentrenamiento Diario desde S3:** Un script integrado descarga datos históricos de ventas desde AWS S3, corre el pipeline de Machine Learning y sube opcionalmente los modelos entrenados.
* **Formato de Modelos:** Los modelos se serializan nativamente usando `joblib` para una carga y predicción local ultrarrápida.

---

## Estructura del Proyecto

```
ml-native/
├── README.md                  # Este archivo de documentación
├── requirements.txt           # Dependencias locales necesarias
├── run_pipeline.py            # Orquestador del pipeline local (datos simulados)
├── run_retrain_s3.py          # Script programable para reentrenamiento diario desde S3
├── data/
│   ├── raw/                   # Datos crudos locales (e.g. reservas.csv)
│   └── processed/             # Características e imputadores persistidos
├── models/                    # Modelos de Machine Learning entrenados (.joblib)
├── src/                       # Código fuente del preprocesamiento y entrenamiento
│   ├── __init__.py
│   ├── generate_dataset.py    # Generador de datos sintéticos
│   ├── feature_engineering.py # Transformación de variables e imputación cold-start
│   ├── train_modelo_a.py      # Entrenamiento del regresor de Mozo
│   ├── train_modelo_b.py      # Entrenamiento de los clasificadores de Menú
│   ├── evaluate.py            # Evaluación cuantitativa y comparación de versiones
│   └── inference.py           # Orquestador de predicción (mozos y platos)
└── tests/                     # Tests unitarios locales
    ├── __init__.py
    ├── test_features.py       # Valida feature engineering sin data leakage
    └── test_inference.py      # Valida consistencia en el formato de salida
```

---

## Cómo Ejecutar Localmente

### 1. Instalar Dependencias
Se recomienda utilizar un entorno virtual de Python (>= 3.9):
```bash
pip install -r requirements.txt
```

### 2. Correr Pipeline con Datos Simulados
Este comando genera 10.000 reservas ficticias, procesa las variables, entrena todos los modelos e imprime métricas de prueba:
```bash
python run_pipeline.py
```

### 3. Ejecutar los Tests Unitarios
Para comprobar que los datos, el preprocesamiento y las predicciones funcionan de forma robusta y consistente:
```bash
pytest tests/ -v
```

---

## Configuración del Reentrenamiento Diario desde S3

El script `run_retrain_s3.py` está diseñado para programarse diariamente (por ejemplo, mediante una tarea Cron en Linux, Task Scheduler en Windows, o un Container en ECS).

### Requisitos Previos:
Tener credenciales de AWS configuradas en el entorno:
```bash
export AWS_ACCESS_KEY_ID="tu-access-key-id"
export AWS_SECRET_ACCESS_KEY="tu-secret-access-key"
export AWS_DEFAULT_REGION="us-east-1"
```

### Ejecutar el Reentrenamiento Diario:
```bash
python run_retrain_s3.py \
  --bucket "nombre-de-tu-bucket-s3" \
  --key "ruta/en/s3/reservas_daily.csv" \
  --dest "data/raw/reservas.csv" \
  --upload-models
```

### Parámetros Disponibles:
* `--bucket` (Requerido): Nombre del bucket de AWS S3 que aloja los datos.
* `--key` (Requerido): Ruta del archivo CSV en S3 (e.g. `data/reservas_2026.csv`).
* `--dest` (Opcional): Ruta local de almacenamiento del CSV descargado. Por defecto `data/raw/reservas.csv`.
* `--upload-models` (Opcional): Si se proporciona, el script subirá los modelos resultantes de regreso al bucket de S3.
* `--s3-prefix` (Opcional): Ruta de destino en S3 para subir los modelos entrenados. Por defecto `models/daily/`.
