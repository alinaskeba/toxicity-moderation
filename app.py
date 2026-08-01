from pathlib import Path
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os

os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

mlflow.set_tracking_uri("http://localhost:5000")

model = mlflow.sklearn.load_model(
    "models:/toxicity-moderation-model@champion"
)
print(type(model))
print(model)

app = FastAPI()


class PredictionRequest(BaseModel):
    text: str


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    text = request.text.strip()

    if not text:
        return {
            "error": "Text must not be empty"
        }

    prediction = int(
        model.predict([text])[0]
    )

    probability = float(
        model.predict_proba([text])[0][1]
    )

    label = (
        "toxic"
        if prediction == 1
        else "non-toxic"
    )

    return {
        "label": label,
        "label_id": prediction,
        "probability": round(probability, 4),
    }