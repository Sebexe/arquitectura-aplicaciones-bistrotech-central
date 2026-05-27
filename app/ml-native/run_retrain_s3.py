"""
Script de Reentrenamiento Diario desde AWS S3 para BistroTech.

Este script descarga el dataset actualizado desde S3, ejecuta el pipeline
de reentrenamiento nativo de los modelos A (Mozo) y B (Menú), opcionalmente
sube los modelos resultantes de regreso al bucket de S3, y finalmente
actualiza la definición del pipeline de reentrenamiento en SageMaker Pipelines.

Requisitos:
  - boto3 instalado (incluido en requirements.txt)
  - sagemaker instalado (para el upsert del pipeline)
  - Credenciales AWS configuradas en el entorno de ejecución
    (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, o rol de IAM asignado).

Variables de entorno para la integración con SageMaker:
  SAGEMAKER_ROLE_ARN  — ARN del rol de ejecución de SageMaker
  S3_BUCKET           — Nombre del bucket S3 (se infiere del --bucket si no está seteado)

Ejemplo de ejecución diaria programada:
  python run_retrain_s3.py --bucket "mi-bucket-bistrotech" --key "data/raw/reservas_daily.csv" --upload-models
"""
import argparse
import datetime
import logging
import os
import sys
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def download_from_s3(bucket: str, key: str, dest_path: str):
    """Descarga el archivo CSV con los datos de entrenamiento desde S3."""
    logger.info("Iniciando descarga desde S3 — Bucket: '%s', Clave: '%s'...", bucket, key)
    s3 = boto3.client("s3")
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        s3.download_file(bucket, key, dest_path)
        logger.info("[OK] Descarga completa en '%s'.", dest_path)
    except NoCredentialsError:
        logger.error("[ERR] Credenciales de AWS no encontradas. Configura tus variables de entorno o rol de IAM.")
        raise
    except ClientError as e:
        logger.error("[ERR] Error de cliente AWS al descargar: %s", e)
        raise
    except Exception as e:
        logger.error("[ERR] Error inesperado al descargar de S3: %s", e)
        raise


def upload_to_s3(bucket: str, local_file: str, s3_key: str):
    """Sube un archivo local de modelo a S3."""
    s3 = boto3.client("s3")
    try:
        s3.upload_file(local_file, bucket, s3_key)
        logger.info("[OK] Sube exitosa: '%s' -> S3://%s/%s", local_file, bucket, s3_key)
    except Exception as e:
        logger.error("[ERR] Falló la subida de '%s' a S3: %s", local_file, e)


def execute_pipeline(raw_csv_path: str):
    """Corre todo el pipeline de entrenamiento local."""
    from src.feature_engineering import build_features, save_processed
    from src.train_modelo_a import train as train_modelo_a
    from src.train_modelo_b import train_all as train_modelo_b
    import pandas as pd

    # 1. Feature Engineering
    logger.info("Paso 1: Ejecutando Feature Engineering...")
    df = pd.read_csv(raw_csv_path)
    X, targets = build_features(df)
    save_processed(X, targets)

    # 2. Entrenar Modelo A (Afinidad de Mozo)
    logger.info("Paso 2: Reentrenando Modelo A...")
    metrics_a = train_modelo_a(raw_csv_path)
    logger.info("Modelo A reentrenado con éxito. RMSE: %.4f", metrics_a["rmse"])

    # 3. Entrenar Modelos B (Menú)
    logger.info("Paso 3: Reentrenando Modelos B...")
    metrics_b = train_modelo_b(raw_csv_path)
    logger.info("Modelos B reentrenados con éxito.")

    # 4. Actualizar versión
    os.makedirs("models", exist_ok=True)
    today_str = datetime.date.today().strftime("%Y%m%d")
    version_str = f"v_native_{today_str}"
    version_file = "models/version.txt"
    with open(version_file, "w") as f:
        f.write(version_str)
    logger.info("Versión del modelo actualizada a '%s'", version_str)

    return version_str


def upload_all_models(bucket: str, prefix: str):
    """Localiza todos los archivos de modelo en la carpeta models/ y los sube a S3."""
    logger.info("Iniciando subida de artefactos a S3 (Prefijo: '%s')...", prefix)
    
    # Lista de archivos a subir
    files_to_upload = [
        "models/modelo_a_mozo.joblib",
        "models/feature_names_a.json",
        "models/modelo_b_entrada.joblib",
        "models/label_encoder_entrada.joblib",
        "models/modelo_b_principal.joblib",
        "models/label_encoder_principal.joblib",
        "models/modelo_b_postre.joblib",
        "models/label_encoder_postre.joblib",
        "models/modelo_b_bebida.joblib",
        "models/label_encoder_bebida.joblib",
        "models/version.txt",
        # Preprocesador
        "data/processed/preprocessor.joblib",
        "data/processed/preprocessor.json"
    ]

    for local_file in files_to_upload:
        if os.path.exists(local_file):
            filename = os.path.basename(local_file)
            # Mantener subcarpetas si es preprocessor
            if "preprocessor" in local_file:
                s3_key = os.path.join(prefix, "processed", filename).replace("\\", "/")
            else:
                s3_key = os.path.join(prefix, filename).replace("\\", "/")
            
            upload_to_s3(bucket, local_file, s3_key)
        else:
            logger.warning("Archivo esperado no encontrado para subir: '%s'", local_file)


def upsert_sagemaker_pipeline(bucket: str):
    """
    Registra o actualiza el pipeline de reentrenamiento en SageMaker.

    Usa SAGEMAKER_ROLE_ARN y S3_BUCKET del entorno. Si S3_BUCKET no está
    configurado explicitamente, usa el bucket del argumento --bucket.
    Omite silenciosamente si SAGEMAKER_ROLE_ARN no está configurado.
    """
    # Propagar el bucket al env si no estaba definido
    if not os.environ.get("S3_BUCKET"):
        os.environ["S3_BUCKET"] = bucket

    from src.sagemaker_pipeline import upsert_pipeline_if_configured
    try:
        arn = upsert_pipeline_if_configured()
        if arn:
            logger.info("[OK] SageMaker Pipeline actualizado: %s", arn)
        else:
            logger.info("[SKIP] Upsert de SageMaker Pipeline omitido (SAGEMAKER_ROLE_ARN no configurado).")
    except Exception as e:
        logger.error("[ERR] Error al actualizar SageMaker Pipeline: %s", e)
        raise


def main():
    parser = argparse.ArgumentParser(description="Reentrenamiento diario BistroTech usando AWS S3.")
    parser.add_argument("--bucket", type=str, required=True, help="Nombre del bucket de S3")
    parser.add_argument("--key", type=str, required=True, help="Ruta/Clave del CSV de datos en S3")
    parser.add_argument("--dest", type=str, default="data/raw/reservas.csv", help="Ruta local destino para el CSV")
    parser.add_argument("--upload-models", action="store_true", help="Si se activa, subirá los modelos reentrenados a S3")
    parser.add_argument("--s3-prefix", type=str, default="models/daily/", help="Prefijo/Directorio en S3 para subir los modelos")
    parser.add_argument("--skip-sagemaker-upsert", action="store_true", help="Si se activa, omitir el upsert del SageMaker Pipeline")

    args = parser.parse_args()

    try:
        # 1. Descarga del dataset
        download_from_s3(args.bucket, args.key, args.dest)

        # 2. Corrida de entrenamiento
        version = execute_pipeline(args.dest)

        # 3. Subida opcional a S3
        if args.upload_models:
            # Añadir versión al prefijo para no sobrescribir sin control
            s3_prefix_versioned = os.path.join(args.s3_prefix, version).replace("\\", "/")
            upload_all_models(args.bucket, s3_prefix_versioned)

        # 4. Actualizar el pipeline de reentrenamiento en SageMaker
        if not args.skip_sagemaker_upsert:
            upsert_sagemaker_pipeline(args.bucket)

        logger.info("[OK] Reentrenamiento diario completado exitosamente.")

    except Exception as e:
        logger.critical("[ERR] El reentrenamiento falló: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
