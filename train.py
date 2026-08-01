import json
from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB


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


def build_model(model_type: str) -> Pipeline:

    if model_type == "logistic":
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    elif model_type == "naive_bayes":
        classifier = MultinomialNB()

    else:
        raise ValueError(f"Unknown model: {model_type}")

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                ),
            ),
            (
                "classifier",
                classifier,
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
        "accuracy": float(
            accuracy_score(y_test, y_pred)
        ),
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
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("toxicity-moderation")
    MODEL_NAME = "toxicity-moderation-model"
    client = MlflowClient()

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

    models = {
        "LogisticRegression": "logistic",
        "NaiveBayes": "naive_bayes",
    }

    best_model = None
    best_metrics = None
    best_name = None
    best_f1 = -1
    best_version = None

    for model_name, model_type in models.items():

        model = build_model(model_type)

        with mlflow.start_run(run_name=model_name):

            mlflow.log_param("algorithm", model_name)
            mlflow.log_param("vectorizer", "TF-IDF")
            mlflow.log_param("max_features", 50000)
            mlflow.log_param("ngram_range", "(1,2)")
            mlflow.log_param("min_df", 2)
            mlflow.log_param("max_df", 0.95)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)

            print(f"\nTraining {model_name}...")

            model.fit(x_train, y_train)

            metrics = evaluate_model(
                model,
                x_test,
                y_test,
            )

            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("precision_toxic", metrics["precision_toxic"])
            mlflow.log_metric("recall_toxic", metrics["recall_toxic"])
            mlflow.log_metric("f1_toxic", metrics["f1_toxic"])
            mlflow.log_metric("roc_auc", metrics["roc_auc"])

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name=MODEL_NAME,
            )
            latest = client.search_model_versions(
                f"name='{MODEL_NAME}'"
            )

            version = max(
            latest,
            key=lambda m: int(m.version),
            ).version

            print(f"Registered model version: {version}")

            save_artifacts(
                model,
                metrics,
            )

            mlflow.log_artifact(str(METRICS_PATH))

            if metrics["f1_toxic"] > best_f1:
                best_f1 = metrics["f1_toxic"]
                best_model = model
                best_metrics = metrics
                best_name = model_name
                best_version = version

    print(f"Champion model: {best_name}")
    print(f"Champion F1-score: {best_f1:.4f}")

    client.set_registered_model_alias(
    MODEL_NAME,
    "champion",
    best_version,
    )

    print(
        f"Champion alias assigned to version {best_version}"
    )

    save_artifacts(
        best_model,
        best_metrics,
    )

if __name__ == "__main__":
    main()