"""
Módulo de integración con SageMaker Pipelines para BistroTech (ml-native).

Construye y hace upsert del pipeline de reentrenamiento en AWS SageMaker
usando los scripts nativos de ml-native como código fuente del contenedor.

El pipeline tiene dos pasos:
  1. ProcessingStep: feature engineering (src/feature_engineering.py)
  2. TrainingStep:   entrenamiento de Modelo A y B (src/train_modelo_a.py + src/train_modelo_b.py)

Variables de entorno requeridas:
  SAGEMAKER_ROLE_ARN  — ARN del rol de ejecución de SageMaker
  S3_BUCKET           — Nombre del bucket S3 de ML (ej: bistrotech-ml-<account>-<region>)
  AWS_REGION          — Región de AWS (default: us-east-1)
  SM_INSTANCE_TYPE    — Tipo de instancia para processing/training (default: ml.m5.large)
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

PIPELINE_NAME = "bistrotech-native-retrain-pipeline"
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "")
S3_BUCKET = os.environ.get("S3_BUCKET", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
# Para procesamiento (Paso 1) dejamos la t3.medium que ya vimos que funciona
PROCESSING_INSTANCE_TYPE = os.environ.get("SM_PROCESSING_INSTANCE_TYPE", "ml.t3.medium")

# Para entrenamiento (Paso 2) probamos la m5.xlarge que suele tener cuota libre
TRAINING_INSTANCE_TYPE = os.environ.get("SM_TRAINING_INSTANCE_TYPE", "ml.m5.xlarge")
MODEL_PACKAGE_GROUP_NAME = os.environ.get(
    "SM_MODEL_PACKAGE_GROUP_NAME", "bistrotech-native-modelos"
)
ENDPOINT_NAME = os.environ.get("SM_ENDPOINT_NAME", "bistrotech-native-endpoint")
# Imágenes ECR gestionadas por AWS para sklearn y xgboost
_SKLEARN_IMAGE = f"683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
_XGB_IMAGE = f"683313688378.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-xgboost:1.7-1"


def _check_env() -> bool:
    """Valida que las variables de entorno mínimas estén configuradas."""
    if not SAGEMAKER_ROLE_ARN:
        logger.warning(
            "SAGEMAKER_ROLE_ARN no configurada. El upsert del pipeline se omitirá."
        )
        return False
    if not S3_BUCKET:
        logger.warning(
            "S3_BUCKET no configurada. El upsert del pipeline se omitirá."
        )
        return False
    return True


def build_pipeline():
    """
    Construye el SageMaker Pipeline usando los scripts nativos de ml-native.

    El código fuente (src/) se sube automáticamente al bucket S3 como parte
    del SourceDir del ScriptProcessor/Estimator.

    Returns:
        sagemaker.workflow.pipeline.Pipeline listo para .upsert()
    """
    try:
        import sagemaker
        from sagemaker.workflow.pipeline import Pipeline
        from sagemaker.workflow.steps import ProcessingStep, TrainingStep
        from sagemaker.workflow.condition_step import ConditionStep
        from sagemaker.workflow.conditions import ConditionLessThan, ConditionGreaterThan
        from sagemaker.workflow.properties import PropertyFile
        from sagemaker.workflow.functions import JsonGet
        from sagemaker.workflow.step_collections import RegisterModel
        from sagemaker.workflow.parameters import ParameterString
        from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
        from sagemaker.estimator import Estimator
        from sagemaker.model import Model
        from sagemaker.inputs import TrainingInput
        from sagemaker.model_metrics import MetricsSource, ModelMetrics
    except ImportError as exc:
        raise ImportError(
            f"Error al importar el SDK de SageMaker: {exc}. "
            "Asegúrate de instalar: pip install sagemaker setuptools"
        ) from exc

    sess = sagemaker.Session()

    # Parámetro del pipeline: URI del CSV de entrada en S3
    input_data_uri = ParameterString(
        name="InputDataUri",
        default_value=f"s3://{S3_BUCKET}/data/raw/reservas.csv",
    )
    endpoint_name = ParameterString(
        name="EndpointName",
        default_value=ENDPOINT_NAME,
    )

    # ── Step 1: Feature Engineering ──────────────────────────────────────────
    processor = ScriptProcessor(
        image_uri=_SKLEARN_IMAGE,
        command=["python3"],
        instance_type=PROCESSING_INSTANCE_TYPE,
        instance_count=1,
        role=SAGEMAKER_ROLE_ARN,
        sagemaker_session=sess,
        base_job_name="bistrotech-features",
    )

    step_features = ProcessingStep(
        name="FeatureEngineering",
        processor=processor,
        # El script de feature engineering se sube desde el directorio local src/
        code="src/feature_engineering.py",
        inputs=[
            ProcessingInput(
                source=input_data_uri,
                destination="/opt/ml/processing/input",
                input_name="raw-data",
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=f"s3://{S3_BUCKET}/data/processed/",
                output_name="processed-data",
            )
        ],
        job_arguments=[
            "--input-path", "/opt/ml/processing/input/reservas.csv",
            "--output-path", "/opt/ml/processing/output",
        ],
    )

    # ── Step 2: Training (Modelo A + B) ──────────────────────────────────────
    estimator = Estimator(
        image_uri=_XGB_IMAGE,
        role=SAGEMAKER_ROLE_ARN,
        instance_type=TRAINING_INSTANCE_TYPE,
        instance_count=1,
        output_path=f"s3://{S3_BUCKET}/models/",
        sagemaker_session=sess,
        base_job_name="bistrotech-train",
        # En contenedores framework de SageMaker, entry_point debe ser relativo a source_dir
        # para evitar resolución inválida del módulo (src/train_modelo_a).
        entry_point="train_modelo_a.py",
        # Incluye todos los módulos del paquete de entrenamiento.
        source_dir="src",
        hyperparameters={
            "n-estimators": 300,
            "max-depth": 6,
        },
    )

    step_train = TrainingStep(
        name="TrainModelos",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_features.properties.ProcessingOutputConfig.Outputs[
                    "processed-data"
                ].S3Output.S3Uri,
                content_type="text/csv",
            )
        },
        depends_on=[step_features],
    )

    # ── Step 3: Baseline Metrics (modelo aprobado actual) ────────────────────
    baseline_processor = ScriptProcessor(
        image_uri=_SKLEARN_IMAGE,
        command=["python3"],
        instance_type=PROCESSING_INSTANCE_TYPE,
        instance_count=1,
        role=SAGEMAKER_ROLE_ARN,
        sagemaker_session=sess,
        base_job_name="bistrotech-baseline-metrics",
    )
    baseline_metrics_property = PropertyFile(
        name="BaselineMetrics",
        output_name="baseline",
        path="baseline_metrics.json",
    )
    step_baseline = ProcessingStep(
        name="GetBaselineMetrics",
        processor=baseline_processor,
        code="src/get_baseline_metrics.py",
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=f"s3://{S3_BUCKET}/metrics/baseline/",
                output_name="baseline",
            )
        ],
        job_arguments=[
            "--model-package-group-name",
            MODEL_PACKAGE_GROUP_NAME,
            "--output-path",
            "/opt/ml/processing/output/baseline_metrics.json",
        ],
        property_files=[baseline_metrics_property],
        depends_on=[step_train],
    )

    # ── Step 4: Evaluación del candidato ─────────────────────────────────────
    eval_processor = ScriptProcessor(
        image_uri=_SKLEARN_IMAGE,
        command=["python3"],
        instance_type=PROCESSING_INSTANCE_TYPE,
        instance_count=1,
        role=SAGEMAKER_ROLE_ARN,
        sagemaker_session=sess,
        base_job_name="bistrotech-eval-candidate",
    )
    candidate_metrics_property = PropertyFile(
        name="CandidateMetrics",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_evaluate = ProcessingStep(
        name="EvaluateCandidate",
        processor=eval_processor,
        code="src/evaluate_candidate.py",
        inputs=[
            ProcessingInput(
                source=step_features.properties.ProcessingOutputConfig.Outputs[
                    "processed-data"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/processed",
                input_name="processed",
            ),
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
                input_name="model-artifact",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{S3_BUCKET}/metrics/candidate/",
                output_name="evaluation",
            )
        ],
        job_arguments=[
            "--model-artifact-path",
            "/opt/ml/processing/model/model.tar.gz",
            "--processed-dir",
            "/opt/ml/processing/processed",
            "--evaluation-output-path",
            "/opt/ml/processing/evaluation/evaluation.json",
            "--registry-output-path",
            "/opt/ml/processing/evaluation/model_quality.json",
        ],
        property_files=[candidate_metrics_property],
        depends_on=[step_baseline],
    )

    # ── Step 5: Registro condicional (solo si mejora) ───────────────────────
    trained_model = Model(
        image_uri=_XGB_IMAGE,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=SAGEMAKER_ROLE_ARN,
        sagemaker_session=sess,
    )
    model_metrics = ModelMetrics(
        model_quality=MetricsSource(
            s3_uri=step_evaluate.properties.ProcessingOutputConfig.Outputs[
                "evaluation"
            ].S3Output.S3Uri
            + "/model_quality.json",
            content_type="application/json",
        ),
    )
    step_register = RegisterModel(
        name="RegisterModeloA",
        model=trained_model,
        content_types=["application/json", "text/csv"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        approval_status="Approved",
        model_metrics=model_metrics,
        description="BistroTech ml-native Modelo A (XGBoost) entrenado por SageMaker Pipeline.",
        depends_on=[step_evaluate],
    )

    # ── Step 6: Deploy condicional ───────────────────────────────────────────
    deploy_processor = ScriptProcessor(
        image_uri=_SKLEARN_IMAGE,
        command=["python3"],
        instance_type=PROCESSING_INSTANCE_TYPE,
        instance_count=1,
        role=SAGEMAKER_ROLE_ARN,
        sagemaker_session=sess,
        base_job_name="bistrotech-deploy-endpoint",
    )
    step_deploy = ProcessingStep(
        name="DeployEndpoint",
        processor=deploy_processor,
        code="src/deploy_endpoint.py",
        job_arguments=[
            "--endpoint-name",
            endpoint_name,
            "--model-data-url",
            step_train.properties.ModelArtifacts.S3ModelArtifacts,
            "--image-uri",
            _XGB_IMAGE,
            "--role-arn",
            SAGEMAKER_ROLE_ARN,
            "--instance-type",
            "ml.m5.large",
        ],
        depends_on=[step_register],
    )
    step_approve_and_deploy = ConditionStep(
        name="ApproveAndDeployIfBetter",
        conditions=[
            ConditionLessThan(
                left=JsonGet(
                    step_name=step_evaluate.name,
                    property_file=candidate_metrics_property,
                    json_path="metrics.rmse",
                ),
                right=JsonGet(
                    step_name=step_baseline.name,
                    property_file=baseline_metrics_property,
                    json_path="metrics.rmse",
                ),
            ),
            ConditionLessThan(
                left=JsonGet(
                    step_name=step_evaluate.name,
                    property_file=candidate_metrics_property,
                    json_path="metrics.mae",
                ),
                right=JsonGet(
                    step_name=step_baseline.name,
                    property_file=baseline_metrics_property,
                    json_path="metrics.mae",
                ),
            ),
            ConditionGreaterThan(
                left=JsonGet(
                    step_name=step_evaluate.name,
                    property_file=candidate_metrics_property,
                    json_path="metrics.pearson",
                ),
                right=JsonGet(
                    step_name=step_baseline.name,
                    property_file=baseline_metrics_property,
                    json_path="metrics.pearson",
                ),
            ),
        ],
        if_steps=[step_register, step_deploy],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[input_data_uri, endpoint_name],
        steps=[step_features, step_train, step_baseline, step_evaluate, step_approve_and_deploy],
        sagemaker_session=sess,
    )
    return pipeline


def upsert_pipeline() -> str:
    """
    Crea o actualiza el pipeline en SageMaker.

    Returns:
        ARN del pipeline registrado.
    Raises:
        RuntimeError si las variables de entorno no están configuradas.
        ImportError  si el SDK de SageMaker no está instalado.
    """
    if not _check_env():
        raise RuntimeError(
            "Variables de entorno SAGEMAKER_ROLE_ARN y S3_BUCKET son requeridas "
            "para hacer upsert del pipeline."
        )

    logger.info("Construyendo pipeline '%s'...", PIPELINE_NAME)
    pipeline = build_pipeline()

    logger.info("Haciendo upsert del pipeline en SageMaker...")
    response = pipeline.upsert(role_arn=SAGEMAKER_ROLE_ARN)
    pipeline_arn = response["PipelineArn"]
    logger.info("[OK] Pipeline registrado/actualizado: %s", pipeline_arn)
    return pipeline_arn


def upsert_pipeline_if_configured() -> str | None:
    """
    Intenta hacer upsert del pipeline en SageMaker.
    Si las variables de entorno no están configuradas, emite un warning y retorna None
    en lugar de lanzar excepción — útil para runs locales sin AWS.

    Returns:
        ARN del pipeline si tuvo éxito, None si se omitió.
    """
    if not _check_env():
        return None
    try:
        return upsert_pipeline()
    except Exception as exc:
        logger.error("[ERR] No se pudo hacer upsert del pipeline en SageMaker: %s", exc)
        raise


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        arn = upsert_pipeline()
        print(f"Pipeline ARN: {arn}")
    except RuntimeError as e:
        logger.warning("%s", e)
        sys.exit(0)
