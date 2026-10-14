
from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "historical_tickets.csv"

MODELS_DIR = PROJECT_ROOT / "ml" / "models"
RESULTS_DIR = PROJECT_ROOT / "ml" / "results" / "v2_logistic_regression_optimized"

MODEL_PATH = MODELS_DIR / "tfidf_logistic_regression_v2_optimized.joblib"
PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"
REPORT_PATH = RESULTS_DIR / "classification_report.txt"
METRICS_PATH = RESULTS_DIR / "model_metrics.json"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
GRID_RESULTS_PATH = RESULTS_DIR / "grid_search_results.csv"
BEST_PARAMS_PATH = RESULTS_DIR / "best_parameters.json"


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

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    dataframe = dataframe.dropna(
        subset=["customer_message", "category"]
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

    dataframe = dataframe[
        (dataframe["customer_message"] != "") &
        (dataframe["category"] != "")
    ].copy()

    return dataframe


def build_pipeline() -> Pipeline:
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ])


def build_parameter_grid() -> list:
    return [
        {
            "tfidf__analyzer": ["word"],
            "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
            "tfidf__min_df": [1, 2],
            "tfidf__sublinear_tf": [True, False],
            "classifier__C": [0.01, 0.1, 1, 10, 100]
        },
        {
            "tfidf__analyzer": ["char_wb"],
            "tfidf__ngram_range": [(2, 4), (3, 5), (3, 6)],
            "tfidf__min_df": [1, 2],
            "tfidf__sublinear_tf": [True, False],
            "classifier__C": [0.01, 0.1, 1, 10, 100]
        }
    ]


def main():

    print("StayWise V2 - Optimized TF-IDF + Logistic Regression")

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe = load_dataset()

    print(f"\nLoaded historical tickets: {len(dataframe)}")

    print("\nCategory distribution:")
    print(dataframe["category"].value_counts().sort_index())

    X = dataframe["customer_message"]
    y = dataframe["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    pipeline = build_pipeline()
    parameter_grid = build_parameter_grid()

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=cross_validation,
        refit=True,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    print("\nStarting GridSearchCV...")
    print("Optimization metric: Macro F1-score")
    print("Cross-validation: Stratified 5-Fold")

    grid_search.fit(
        X_train,
        y_train
    )

    best_model = grid_search.best_estimator_
    best_parameters = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print("\nGRID SEARCH RESULTS")
    print(f"\nBest CV Macro F1: {best_cv_score:.4f}")

    print("\nBest parameters:")
    for parameter, value in best_parameters.items():
        print(f"{parameter}: {value}")

    predictions = best_model.predict(X_test)

    probabilities = best_model.predict_proba(X_test)
    confidences = probabilities.max(axis=1)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\nHOLDOUT EVALUATION")
    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Macro F1-score: {macro_f1:.4f}")

    report_text = classification_report(
        y_test,
        predictions,
        digits=4,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report_text)

    labels = sorted(y.unique())

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    ax.set_title(
        "StayWise V2 Optimized - Confusion Matrix"
    )

    plt.xticks(rotation=25, ha="right")
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
        file.write(report_text)

    grid_results = pd.DataFrame(
        grid_search.cv_results_
    )

    grid_results = grid_results.sort_values(
        by="rank_test_score"
    )

    grid_results.to_csv(
        GRID_RESULTS_PATH,
        index=False,
        encoding="utf-8"
    )

    best_parameters_serializable = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in best_parameters.items()
    }

    with open(
        BEST_PARAMS_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "best_cv_macro_f1": float(best_cv_score),
                "best_parameters": best_parameters_serializable
            },
            file,
            indent=4,
            ensure_ascii=False
        )

    best_index = grid_search.best_index_

    metrics = {
        "dataset_size": len(dataframe),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "best_cv_macro_f1": float(best_cv_score),
        "best_cv_std": float(
            grid_search.cv_results_["std_test_score"][best_index]
        ),
        "model": "Logistic Regression",
        "features": "Optimized TF-IDF",
        "optimization": "GridSearchCV",
        "scoring": "f1_macro",
        "cv_folds": 5,
        "random_state": 42,
        "classes": labels,
        "best_parameters": best_parameters_serializable
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
        best_model,
        MODEL_PATH
    )

    print("\nSaved files:")
    print(f"Model:            {MODEL_PATH}")
    print(f"Predictions:      {PREDICTIONS_PATH}")
    print(f"Report:           {REPORT_PATH}")
    print(f"Metrics:          {METRICS_PATH}")
    print(f"Confusion matrix: {CONFUSION_MATRIX_PATH}")
    print(f"Grid results:     {GRID_RESULTS_PATH}")
    print(f"Best parameters:  {BEST_PARAMS_PATH}")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
