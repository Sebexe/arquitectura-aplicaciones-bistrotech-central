"""
Evalúa el modelo candidato entrenado en el pipeline y genera reportes JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tarfile
import tempfile

import joblib

from evaluate import evaluate_modelo_a

logger = logging.getLogger(__name__)


def _load_model_from_artifact(model_artifact_path: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        with tarfile.open(model_artifact_path, "r:gz") as tar:
            tar.extractall(tmp_dir)

        model_path = os.path.join(tmp_dir, "modelo_a_mozo.joblib")
        features_path = os.path.join(tmp_dir, "feature_names_a.json")
        model = joblib.load(model_path)
        with open(features_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
        return model, feature_names


def _load_eval_data(processed_dir: str):
    x_path = os.path.join(processed_dir, "X.joblib")
    targets_path = os.path.join(processed_dir, "targets.joblib")
    x_base = joblib.load(x_path)
    targets = joblib.load(targets_path)
    y = targets["propina_rate"]
    x_eval = x_base.loc[y.index].copy()
    return x_eval, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifact-path", default="/opt/ml/processing/model/model.tar.gz")
    parser.add_argument("--processed-dir", default="/opt/ml/processing/processed")
    parser.add_argument(
        "--evaluation-output-path",
        default="/opt/ml/processing/evaluation/evaluation.json",
    )
    parser.add_argument(
        "--registry-output-path",
        default="/opt/ml/processing/evaluation/model_quality.json",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model, feature_names = _load_model_from_artifact(args.model_artifact_path)
    x_eval, y_eval = _load_eval_data(args.processed_dir)
    x_eval = x_eval.reindex(columns=feature_names, fill_value=0)
    metrics = evaluate_modelo_a(model, x_eval, y_eval.values)

    evaluation_report = {"metrics": metrics}
    model_quality_report = {
        "version": 0,
        "dataset_type": "test",
        "dataset": {"name": "bistrotech-processed-holdout"},
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(args.evaluation_output_path), exist_ok=True)
    with open(args.evaluation_output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    os.makedirs(os.path.dirname(args.registry_output_path), exist_ok=True)
    with open(args.registry_output_path, "w", encoding="utf-8") as f:
        json.dump(model_quality_report, f, indent=2)

    logger.info("Reporte evaluación guardado en %s", args.evaluation_output_path)
    logger.info("Reporte model quality guardado en %s", args.registry_output_path)
