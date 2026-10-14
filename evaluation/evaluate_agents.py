from pathlib import Path
from uuid import uuid4

import pandas as pd

from workflow.graph import build_graph


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_CASES_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "test_cases.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "agent_evaluation.csv"
)


def determine_actual_action(result: dict) -> str:
    if "__interrupt__" in result:
        return "human_approval"

    category = result.get("category")

    if category == "Urgent Problem" and result.get("ticket"):
        return "escalate"

    if category == "Unclear":
        return "request_details"

    if result.get("reservation") is None:
        return "request_details"

    return "respond"


def evaluate_test_case(
    graph,
    row: pd.Series
) -> dict:
    thread_id = str(uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    initial_state = {
        "customer_message": row["message"],
        "request_date": row["request_date"],
        "trace": []
    }

    try:
        result = graph.invoke(
            initial_state,
            config=config
        )

        actual_category = result.get(
            "category",
            "Unknown"
        )

        actual_action = determine_actual_action(
            result
        )

        final_response = result.get(
            "final_response",
            ""
        )

        category_correct = (
            actual_category
            == row["expected_category"]
        )

        action_correct = (
            actual_action
            == row["expected_action"]
        )

        passed = (
            category_correct
            and action_correct
        )

        trace = " | ".join(
            result.get("trace", [])
        )

        return {
            "test_id": row["test_id"],
            "message": row["message"],
            "expected_category": row["expected_category"],
            "actual_category": actual_category,
            "category_correct": category_correct,
            "expected_action": row["expected_action"],
            "actual_action": actual_action,
            "action_correct": action_correct,
            "passed": passed,
            "final_response": final_response,
            "trace": trace,
            "error": ""
        }

    except Exception as error:
        return {
            "test_id": row["test_id"],
            "message": row["message"],
            "expected_category": row["expected_category"],
            "actual_category": "ERROR",
            "category_correct": False,
            "expected_action": row["expected_action"],
            "actual_action": "ERROR",
            "action_correct": False,
            "passed": False,
            "final_response": "",
            "trace": "",
            "error": str(error)
        }


def print_test_result(result: dict) -> None:
    status = (
        "PASS"
        if result["passed"]
        else "FAIL"
    )

    print("\n" + "-" * 70)

    print(
        f"{result['test_id']} → {status}"
    )

    print(
        f"Expected category: "
        f"{result['expected_category']}"
    )

    print(
        f"Actual category:   "
        f"{result['actual_category']}"
    )

    print(
        f"Expected action:   "
        f"{result['expected_action']}"
    )

    print(
        f"Actual action:     "
        f"{result['actual_action']}"
    )

    if result["trace"]:
        print(
            f"Trace:             "
            f"{result['trace']}"
        )

    if result["error"]:
        print(
            f"Error:             "
            f"{result['error']}"
        )


def print_summary(
    results_df: pd.DataFrame
) -> None:
    total = len(results_df)

    correct_categories = int(
        results_df["category_correct"].sum()
    )

    correct_actions = int(
        results_df["action_correct"].sum()
    )

    passed = int(
        results_df["passed"].sum()
    )

    classification_accuracy = (
        correct_categories / total
    )

    action_accuracy = (
        correct_actions / total
    )

    overall_accuracy = (
        passed / total
    )

    print("\n")
    print("=" * 70)
    print("STAYWISE MULTI-AGENT EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Total test cases:          {total}"
    )

    print(
        f"Correct classifications:   "
        f"{correct_categories}/{total}"
    )

    print(
        f"Classification accuracy:   "
        f"{classification_accuracy:.2%}"
    )

    print(
        f"Correct actions:           "
        f"{correct_actions}/{total}"
    )

    print(
        f"Action accuracy:           "
        f"{action_accuracy:.2%}"
    )

    print(
        f"Fully passed cases:        "
        f"{passed}/{total}"
    )

    print(
        f"Overall accuracy:          "
        f"{overall_accuracy:.2%}"
    )

    print("=" * 70)

    if correct_categories >= 16:
        print(
            "✓ Classification target achieved (>= 16/20)."
        )
    else:
        print(
            "✗ Classification target NOT achieved."
        )


def main():
    print("=" * 70)
    print("StayWise Multi-Agent Automatic Evaluation")
    print("=" * 70)

    test_cases = pd.read_csv(
        TEST_CASES_PATH
    )

    print(
        f"\nLoaded {len(test_cases)} test cases."
    )

    graph = build_graph()

    results = []

    for _, row in test_cases.iterrows():
        result = evaluate_test_case(
            graph,
            row
        )

        results.append(
            result
        )

        print_test_result(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
        encoding="utf-8"
    )

    print_summary(
        results_df
    )

    print(
        f"\nDetailed results saved to:\n"
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()