# Toxic Comment Moderation

Course project for MLOps.

Baseline binary classifier for toxic comment detection.

## Assignment 2

Implemented:

- Data preprocessing
- Baseline model training (TF-IDF + Logistic Regression)
- FastAPI API
- Docker container
- MinIO object storage
- Airflow DAG for uploading new training batches

## Technologies

- Python
- scikit-learn
- FastAPI
- Docker
- Apache Airflow
- MinIO

## Model

- TF-IDF + Logistic Regression
- ROC-AUC: **0.97**

## Next steps

The following components will be added in future assignments:

- MLflow tracking
- Model Registry
- Challenger–Champion deployment
- End-to-end training pipeline