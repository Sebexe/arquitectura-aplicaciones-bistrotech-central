"""
Despliega/actualiza endpoint de SageMaker con el modelo candidato.
"""
from __future__ import annotations

import argparse
import logging
import time

import boto3

logger = logging.getLogger(__name__)


def upsert_endpoint(
    endpoint_name: str,
    model_data_url: str,
    image_uri: str,
    role_arn: str,
    instance_type: str,
) -> None:
    sm = boto3.client("sagemaker")
    ts = int(time.time())
    model_name = f"{endpoint_name}-model-{ts}"
    config_name = f"{endpoint_name}-cfg-{ts}"

    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
        },
    )

    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
        logger.info("Endpoint existente actualizado: %s", endpoint_name)
    except sm.exceptions.ClientError:
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
        logger.info("Endpoint creado: %s", endpoint_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--model-data-url", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--instance-type", default="ml.m5.large")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    upsert_endpoint(
        endpoint_name=args.endpoint_name,
        model_data_url=args.model_data_url,
        image_uri=args.image_uri,
        role_arn=args.role_arn,
        instance_type=args.instance_type,
    )
