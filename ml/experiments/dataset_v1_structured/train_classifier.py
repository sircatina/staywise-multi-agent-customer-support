from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ( accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "historical_tickets.csv"

MODELS_DIR = PROJECT_ROOT / "ml" / "models"
RESULTS_DIR = PROJECT_ROOT / "ml" / "results"

MODEL_PATH = MODELS_DIR / "tfidf_logistic_regression.joblib"
PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"
REPORT_PATH = RESULTS_DIR / "classification_report.txt"
METRICS_PATH = RESULTS_DIR / "model_metrics.json"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    dataframe = pd.read_csv(DATA_PATH)

    required_columns = {
        "customer_message",
        "category"
    }

    missing_columns = required_columns - set(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{missing_columns}"
        )

    dataframe = dataframe.dropna(
        subset=[
            "customer_message",
            "category"
        ]
    ).copy()

    dataframe["customer_message"] = (
        dataframe["customer_message"]
        .astype(str)
        .str.strip()
    )

    dataframe["category"] = (
        dataframe["category"]
        .astype(str)
        .str.strip()
    )

    return dataframe


def build_model() -> Pipeline:
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])


def main():
    print("=" * 70)
    print("StayWise Classic ML Classifier Training")
    print("=" * 70)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe = load_dataset()

    print(
        f"\nLoaded historical tickets: "
        f"{len(dataframe)}"
    )

    print("\nCategory distribution:")

    print(
        dataframe["category"]
        .value_counts()
        .sort_index()
    )

    X = dataframe["customer_message"]
    y = dataframe["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples:  {len(X_test)}"
    )

    model = build_model()

    print(
        "\nTraining TF-IDF + "
        "Logistic Regression..."
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    confidences = probabilities.max(
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print(
        f"\nAccuracy: {accuracy:.2%}"
    )

    report_text = classification_report(
        y_test,
        predictions,
        digits=4,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report_text)

    labels = sorted(
        y.unique()
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    ax.set_title(
        "StayWise Classic ML - Confusion Matrix"
    )

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    predictions_dataframe = pd.DataFrame({
        "customer_message": X_test.values,
        "actual_category": y_test.values,
        "predicted_category": predictions,
        "confidence": confidences
    })

    predictions_dataframe["correct"] = (
        predictions_dataframe["actual_category"]
        ==
        predictions_dataframe["predicted_category"]
    )

    predictions_dataframe.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8"
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            report_text
        )

    metrics = {
        "dataset_size": len(dataframe),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "accuracy": float(accuracy),
        "model": "Logistic Regression",
        "features": "TF-IDF unigram + bigram",
        "random_state": 42,
        "classes": labels
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False
        )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\nSaved files:")
    print(f"Model:            {MODEL_PATH}")
    print(f"Predictions:      {PREDICTIONS_PATH}")
    print(f"Report:           {REPORT_PATH}")
    print(f"Metrics:          {METRICS_PATH}")
    print(f"Confusion matrix: {CONFUSION_MATRIX_PATH}")

    print(
        "\nTraining completed successfully."
    )


if __name__ == "__main__":
    main()