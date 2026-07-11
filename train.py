import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/training_data.csv")
MODEL_PATH = Path("models/toxicity_model.joblib")
METRICS_PATH = Path("models/metrics.json")

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_training_data(
    data_path: Path,
) -> tuple[pd.Series, pd.Series]:
    """Read prepared data and return features and target."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path.resolve()}\n"
            "Run 'python prepare_data.py' first."
        )

    df = pd.read_csv(data_path)

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    invalid_labels = set(df["label"].unique()) - {0, 1}

    if invalid_labels:
        raise ValueError(
            f"Invalid labels found: {sorted(invalid_labels)}"
        )

    print(f"Loaded dataset: {df.shape[0]} rows")

    return df["text"], df["label"]


def build_model() -> Pipeline:
    """Create the baseline TF-IDF and Logistic Regression pipeline."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline,
    x_test: pd.Series,
    y_test: pd.Series,
) -> dict:
    """Evaluate the trained model and return serializable metrics."""

    y_pred = model.predict(x_test)
    y_probability = model.predict_proba(x_test)[:, 1]

    report = classification_report(
        y_test,
        y_pred,
        digits=4,
    )

    print("\nClassification report:")
    print(report)

    metrics = {
        "precision_toxic": float(
            precision_score(y_test, y_pred)
        ),
        "recall_toxic": float(
            recall_score(y_test, y_pred)
        ),
        "f1_toxic": float(
            f1_score(y_test, y_pred)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, y_probability)
        ),
        "confusion_matrix": (
            confusion_matrix(y_test, y_pred).tolist()
        ),
        "test_size": int(len(y_test)),
    }

    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Toxic F1: {metrics['f1_toxic']:.4f}")

    return metrics


def save_artifacts(
    model: Pipeline,
    metrics: dict,
) -> None:
    """Save the trained pipeline and evaluation metrics."""

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, MODEL_PATH)

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print(f"\nModel saved to: {MODEL_PATH.resolve()}")
    print(f"Metrics saved to: {METRICS_PATH.resolve()}")


def main() -> None:
    x, y = load_training_data(DATA_PATH)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"Training rows: {len(x_train)}")
    print(f"Testing rows: {len(x_test)}")

    model = build_model()

    print("\nTraining model...")
    model.fit(x_train, y_train)

    print("Training completed.")

    metrics = evaluate_model(
        model,
        x_test,
        y_test,
    )

    save_artifacts(
        model,
        metrics,
    )


if __name__ == "__main__":
    main()