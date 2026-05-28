"""
Obtiene métricas del último modelo aprobado en Model Registry.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from urllib.parse import urlparse

import boto3

logger = logging.getLogger(__name__)

DEFAULT_METRICS = {"rmse": 1.0e9, "mae": 1.0e9, "pearson": -1.0}


def _read_s3_json(s3_uri: str) -> dict:
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def get_latest_approved_metrics(model_package_group_name: str) -> dict:
    sm = boto3.client("sagemaker")
    resp = sm.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    packages = resp.get("ModelPackageSummaryList", [])
    if not packages:
        logger.info("No hay modelo aprobado previo, usando baseline por defecto.")
        return DEFAULT_METRICS

    arn = packages[0]["ModelPackageArn"]
    desc = sm.describe_model_package(ModelPackageName=arn)
    model_metrics = desc.get("ModelMetrics", {})
    quality = model_metrics.get("ModelQuality", {})
    stats = quality.get("Statistics", {})
    s3_uri = stats.get("S3Uri")
    if not s3_uri:
        logger.info("Modelo aprobado sin estadísticas, usando baseline por defecto.")
        return DEFAULT_METRICS

    report = _read_s3_json(s3_uri)
    return {
        "rmse": float(report["metrics"]["rmse"]),
        "mae": float(report["metrics"]["mae"]),
        "pearson": float(report["metrics"]["pearson"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-package-group-name", required=True)
    parser.add_argument("--output-path", default="/opt/ml/processing/output/baseline_metrics.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metrics = get_latest_approved_metrics(args.model_package_group_name)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics}, f, indent=2)
    logger.info("Baseline guardado en %s", args.output_path)
