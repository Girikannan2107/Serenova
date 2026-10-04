from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"

RANDOM_STATE = 42

TARGET = "Outcome"

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

TEST_SIZE = 116
VAL_SIZE = 115


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_data():

    df = pd.read_csv(RAW_PATH)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(np.int64).copy()

    # Treat physiologically implausible zero measurements as missing.
    for column in ZERO_AS_MISSING_COLUMNS:
        X.loc[X[column] == 0, column] = np.nan

    return X, y


# =============================================================================
# MODEL
# =============================================================================

def build_pipeline():

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 80)
    print("SERENOVA — DIABETES RANDOM FOREST TUNING")
    print("=" * 80)

    print(f"Dataset: {RAW_PATH}")

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{RAW_PATH}"
        )

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    X, y = load_data()

    print()
    print(f"Rows: {len(X)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    print("\nTarget distribution:")

    counts = y.value_counts().sort_index()

    for cls in [0, 1]:
        print(
            f"  Class {cls}: {counts.get(cls, 0)}"
        )

    # -------------------------------------------------------------------------
    # LOCKED SPLIT
    #
    # Same random state and sizes as baseline benchmark.
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("1. RECREATE LOCKED SPLIT")
    print("=" * 80)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VAL_SIZE,
        stratify=y_train_full,
        random_state=RANDOM_STATE,
    )

    print(f"Train:      {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test:       {len(X_test)}")

    # Integrity checks.
    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    test_indices = set(X_test.index)

    assert train_indices.isdisjoint(val_indices)
    assert train_indices.isdisjoint(test_indices)
    assert val_indices.isdisjoint(test_indices)

    print("\nRow leakage checks: PASS")

    # -------------------------------------------------------------------------
    # PIPELINE
    # -------------------------------------------------------------------------

    pipeline = build_pipeline()

    # -------------------------------------------------------------------------
    # HYPERPARAMETER SEARCH
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("2. RANDOM FOREST HYPERPARAMETER SEARCH")
    print("=" * 80)

    param_grid = {
        "model__n_estimators": [
            300,
            500,
        ],
        "model__max_depth": [
            None,
            6,
            10,
        ],
        "model__max_features": [
            "sqrt",
            "log2",
            0.8,
        ],
        "model__min_samples_leaf": [
            1,
            2,
            4,
        ],
        "model__min_samples_split": [
            2,
            5,
        ],
        "model__class_weight": [
            "balanced",
            "balanced_subsample",
        ],
    }

    candidate_count = 1

    for values in param_grid.values():
        candidate_count *= len(values)

    print(
        f"Candidate configurations: {candidate_count}"
    )

    print(
        f"Total CV fits: {candidate_count * 5}"
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
        verbose=1,
    )

    search.fit(
        X_train_full,
        y_train_full,
    )

    # -------------------------------------------------------------------------
    # BEST RESULT
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("3. BEST CROSS-VALIDATED MODEL")
    print("=" * 80)

    print(
        f"Best CV Macro F1: "
        f"{search.best_score_:.4f}"
    )

    print("\nBest parameters:")

    for key, value in search.best_params_.items():
        print(
            f"  {key}: {value}"
        )

    # -------------------------------------------------------------------------
    # TOP RESULTS
    # -------------------------------------------------------------------------

    results = pd.DataFrame(
        search.cv_results_
    )

    results = results.sort_values(
        by="mean_test_score",
        ascending=False,
    )

    print()
    print("Top 10 configurations:")

    columns_to_show = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "param_model__n_estimators",
        "param_model__max_depth",
        "param_model__max_features",
        "param_model__min_samples_leaf",
        "param_model__min_samples_split",
        "param_model__class_weight",
    ]

    print(
        results[columns_to_show]
        .head(10)
        .to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # VALIDATION CHECK
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("4. VALIDATION CHECK")
    print("=" * 80)

    tuned_model = search.best_estimator_

    tuned_model.fit(
        X_train,
        y_train,
    )

    val_predictions = tuned_model.predict(
        X_val
    )

    val_f1 = f1_score(
        y_val,
        val_predictions,
        average="macro",
        zero_division=0,
    )

    val_accuracy = accuracy_score(
        y_val,
        val_predictions,
    )

    val_recall = recall_score(
        y_val,
        val_predictions,
        average="macro",
        zero_division=0,
    )

    print(
        f"Validation Accuracy:     {val_accuracy:.4f}"
    )

    print(
        f"Validation Macro Recall: {val_recall:.4f}"
    )

    print(
        f"Validation Macro F1:     {val_f1:.4f}"
    )

    # -------------------------------------------------------------------------
    # LOCKED TEST
    #
    # The test set has NOT participated in GridSearchCV.
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("5. LOCKED TEST EVALUATION")
    print("=" * 80)

    # Fit the selected configuration on train + validation.
    tuned_model.fit(
        X_train_full,
        y_train_full,
    )

    test_predictions = tuned_model.predict(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    test_precision = precision_score(
        y_test,
        test_predictions,
        average="macro",
        zero_division=0,
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        average="macro",
        zero_division=0,
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        average="macro",
        zero_division=0,
    )

    print(
        f"Accuracy:        {test_accuracy:.4f}"
    )

    print(
        f"Macro Precision: {test_precision:.4f}"
    )

    print(
        f"Macro Recall:    {test_recall:.4f}"
    )

    print(
        f"Macro F1:        {test_f1:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=[
                "Outcome 0",
                "Outcome 1",
            ],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")

    print(
        confusion_matrix(
            y_test,
            test_predictions,
        )
    )

    # -------------------------------------------------------------------------
    # BASELINE COMPARISON
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("6. BASELINE VS TUNED")
    print("=" * 80)

    baseline_cv_f1 = 0.7686
    baseline_test_f1 = 0.7413

    print(
        f"Baseline RF CV Macro F1: "
        f"{baseline_cv_f1:.4f}"
    )

    print(
        f"Tuned RF CV Macro F1:    "
        f"{search.best_score_:.4f}"
    )

    print(
        f"CV change: "
        f"{search.best_score_ - baseline_cv_f1:+.4f}"
    )

    print()

    print(
        f"Baseline RF test Macro F1: "
        f"{baseline_test_f1:.4f}"
    )

    print(
        f"Tuned RF test Macro F1:    "
        f"{test_f1:.4f}"
    )

    print(
        f"Test change: "
        f"{test_f1 - baseline_test_f1:+.4f}"
    )

    # -------------------------------------------------------------------------
    # INTEGRITY
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("7. EVALUATION INTEGRITY")
    print("=" * 80)

    print(
        "Test set used during hyperparameter search: NO"
    )

    print(
        "Test set used during preprocessing fitting: NO"
    )

    print(
        "Test set used for model selection: NO"
    )

    print(
        "Test set used only for final evaluation: YES"
    )

    print()
    print(
        "No model artifact saved."
    )

    print(
        "Diabetes Random Forest tuning complete."
    )


if __name__ == "__main__":
    main()