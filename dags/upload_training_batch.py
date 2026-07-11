import os
from pathlib import Path

import boto3
from airflow.sdk import dag, task
from pendulum import datetime


@dag(
    dag_id="upload_training_batch",
    schedule=None,
    start_date=datetime(2026, 7, 1, tz="UTC"),
    catchup=False,
    tags=["data-ingestion"],
)
def upload_training_batch():
    @task
    def upload_to_minio():
        local_file = Path(
            "/opt/airflow/incoming/batch_001.csv"
        )

        if not local_file.exists():
            raise FileNotFoundError(
                f"Batch file not found: {local_file}"
            )

        endpoint = os.environ["MINIO_ENDPOINT"]
        access_key = os.environ["MINIO_ACCESS_KEY"]
        secret_key = os.environ["MINIO_SECRET_KEY"]
        bucket_name = os.environ["MINIO_BUCKET"]

        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        object_name = "incoming/batch_001.csv"

        client.upload_file(
            str(local_file),
            bucket_name,
            object_name,
        )

        print(
            f"Uploaded {local_file} to "
            f"{bucket_name}/{object_name}"
        )

        return object_name

    upload_to_minio()


upload_training_batch()