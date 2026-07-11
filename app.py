from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path("models/toxicity_model.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file not found. Run 'python train.py' first."
    )

model = joblib.load(MODEL_PATH)

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