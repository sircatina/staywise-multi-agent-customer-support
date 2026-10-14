from pathlib import Path
import json
import time

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "tfidf_multinomial_naive_bayes_v2.joblib"
)

TEST_CASES_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "test_cases.csv"
)

AGENT_RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "agent_evaluation.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "v2_naive_bayes"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "comparison_predictions.csv"
)

COMPARISON_PATH = (
    RESULTS_DIR
    / "model_comparison.json"
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model = load_model()

    test_cases = pd.read_csv(
        TEST_CASES_PATH
    )

    messages = test_cases[
        "message"
    ].astype(str)

    expected = test_cases[
        "expected_category"
    ].astype(str)

    start_time = time.perf_counter()

    predictions = model.predict(
        messages
    )

    probabilities = model.predict_proba(
        messages
    )

    confidences = probabilities.max(
        axis=1
    )

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    ml_accuracy = accuracy_score(
        expected,
        predictions
    )

    results = pd.DataFrame({
        "test_id": test_cases[
            "test_id"
        ],
        "message": messages,
        "expected_category": expected,
        "ml_prediction": predictions,
        "ml_confidence": confidences
    })

    results["ml_correct"] = (
        results["expected_category"]
        ==
        results["ml_prediction"]
    )

    agent_accuracy = None

    if AGENT_RESULTS_PATH.exists():
        agent_results = pd.read_csv(
            AGENT_RESULTS_PATH
        )

        agent_results = agent_results[
            [
                "test_id",
                "actual_category",
                "category_correct"
            ]
        ]

        results = results.merge(
            agent_results,
            on="test_id",
            how="left"
        )

        agent_accuracy = (
            agent_results[
                "category_correct"
            ]
            .astype(bool)
            .mean()
        )

    results.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8"
    )

    labels = [
        "Reservation",
        "Payment",
        "Cancellation",
        "Urgent Problem",
        "Unclear"
    ]

    matrix = confusion_matrix(
        expected,
        predictions,
        labels=labels
    )

    report = classification_report(
        expected,
        predictions,
        labels=labels,
        digits=4,
        zero_division=0
    )

    comparison = {
        "test_cases": len(test_cases),
        "ml_accuracy": float(
            ml_accuracy
        ),
        "agent_accuracy": (
            float(agent_accuracy)
            if agent_accuracy is not None
            else None
        ),
        "ml_total_inference_seconds": (
            elapsed_time
        ),
        "ml_average_inference_seconds": (
            elapsed_time
            / len(test_cases)
        ),
        "labels": labels,
        "confusion_matrix": (
            matrix.tolist()
        )
    }

    with open(
        COMPARISON_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            comparison,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nClassic ML accuracy: "
        f"{ml_accuracy:.2%}"
    )

    if agent_accuracy is not None:
        print(
            f"Multi-Agent accuracy: "
            f"{agent_accuracy:.2%}"
        )

    print(
        f"\nML inference time: "
        f"{elapsed_time:.6f} seconds"
    )

    print(
        "\nClassification Report:"
    )

    print(report)

    print(
        "\nPredictions:"
    )

    for _, row in results.iterrows():
        status = (
            "PASS"
            if row["ml_correct"]
            else "FAIL"
        )

        print(
            f"{row['test_id']} → "
            f"{status} | "
            f"Expected: "
            f"{row['expected_category']} | "
            f"ML: "
            f"{row['ml_prediction']} | "
            f"Confidence: "
            f"{row['ml_confidence']:.3f}"
        )

    print(
        f"\nResults saved to: "
        f"{PREDICTIONS_PATH}"
    )

    print(
        f"Comparison saved to: "
        f"{COMPARISON_PATH}"
    )


if __name__ == "__main__":
    main()