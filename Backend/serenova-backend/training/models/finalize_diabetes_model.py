from pathlib import Path
import json

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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"
RESULTS_DIR = BASE_DIR / "data" / "results"

MODEL_PATH = RESULTS_DIR / "diabetes_random_forest_tuned.joblib"
METADATA_PATH = RESULTS_DIR / "diabetes_random_forest_tuned_metadata.json"

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
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 80)
    print("SERENOVA — FINALIZE DIABETES RANDOM FOREST MODEL")
    print("=" * 80)

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{RAW_PATH}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------

    df = pd.read_csv(RAW_PATH)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(np.int64).copy()

    # -------------------------------------------------------------------------
    # SAME PREPROCESSING POLICY AS BENCHMARK
    # -------------------------------------------------------------------------

    for column in ZERO_AS_MISSING_COLUMNS:
        X.loc[X[column] == 0, column] = np.nan

    # -------------------------------------------------------------------------
    # RECREATE EXACT LOCKED SPLIT
    # -------------------------------------------------------------------------

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

    print()
    print("Locked split:")
    print(f"  Train:      {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test:       {len(X_test)}")

    # -------------------------------------------------------------------------
    # INTEGRITY CHECK
    # -------------------------------------------------------------------------

    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    test_indices = set(X_test.index)

    assert train_indices.isdisjoint(val_indices)
    assert train_indices.isdisjoint(test_indices)
    assert val_indices.isdisjoint(test_indices)

    print("\nSplit leakage checks: PASS")

    # -------------------------------------------------------------------------
    # SELECTED TUNED MODEL
    # -------------------------------------------------------------------------

    model = Pipeline(
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
                    n_estimators=500,
                    max_depth=6,
                    max_features="log2",
                    min_samples_leaf=1,
                    min_samples_split=2,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # -------------------------------------------------------------------------
    # FINAL TRAINING
    #
    # Test remains locked.
    # Train + validation are used for final model fitting.
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL TRAINING")
    print("=" * 80)

    print("Training data: train + validation")
    print("Test data: locked and excluded from training")

    model.fit(
        X_train_full,
        y_train_full,
    )

    # -------------------------------------------------------------------------
    # FINAL LOCKED TEST VERIFICATION
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOCKED TEST VERIFICATION")
    print("=" * 80)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Outcome 0",
            "Outcome 1",
        ],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Outcome 0",
                "Outcome 1",
            ],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(matrix)

    # -------------------------------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SAVING ARTIFACT")
    print("=" * 80)

    joblib.dump(
        model,
        MODEL_PATH,
    )

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------

    metadata = {
        "model_name": "Diabetes Random Forest Tuned",
        "task": "binary classification",
        "target": TARGET,
        "target_mapping": {
            "0": "Outcome 0",
            "1": "Outcome 1",
        },
        "features": FEATURE_COLUMNS,
        "zero_as_missing_columns": ZERO_AS_MISSING_COLUMNS,
        "imputation": "median",
        "preprocessing_fitted_inside_pipeline": True,
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 500,
            "max_depth": 6,
            "max_features": "log2",
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "class_weight": "balanced_subsample",
            "random_state": RANDOM_STATE,
        },
        "dataset": {
            "raw_rows": int(len(df)),
            "feature_count": len(FEATURE_COLUMNS),
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
        },
        "selection": {
            "baseline_cv_macro_f1": 0.7686,
            "tuned_cv_macro_f1": 0.773640,
            "cv_improvement": 0.005040,
            "test_macro_f1": float(macro_f1),
            "test_macro_f1_improvement": float(
                macro_f1 - 0.7413
            ),
        },
        "locked_test_metrics": {
            "accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "outcome_0_precision": float(
                report["Outcome 0"]["precision"]
            ),
            "outcome_0_recall": float(
                report["Outcome 0"]["recall"]
            ),
            "outcome_0_f1": float(
                report["Outcome 0"]["f1-score"]
            ),
            "outcome_1_precision": float(
                report["Outcome 1"]["precision"]
            ),
            "outcome_1_recall": float(
                report["Outcome 1"]["recall"]
            ),
            "outcome_1_f1": float(
                report["Outcome 1"]["f1-score"]
            ),
            "confusion_matrix": matrix.tolist(),
        },
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
        "clinical_validation": False,
        "production_approved": False,
        "artifact_status": "experimental",
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # ARTIFACT VERIFICATION
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("ARTIFACT VERIFICATION")
    print("=" * 80)

    print(
        f"Model file exists: "
        f"{'PASS' if MODEL_PATH.exists() else 'FAIL'}"
    )

    print(
        f"Metadata file exists: "
        f"{'PASS' if METADATA_PATH.exists() else 'FAIL'}"
    )

    if MODEL_PATH.exists():
        print(
            f"Model size: "
            f"{MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB"
        )

    if METADATA_PATH.exists():
        print(
            f"Metadata size: "
            f"{METADATA_PATH.stat().st_size / 1024:.2f} KB"
        )

    print()
    print(f"Saved artifact:")
    print(MODEL_PATH)

    print()
    print(f"Saved metadata:")
    print(METADATA_PATH)

    print()
    print("Diabetes tuned Random Forest finalization complete.")

    print()
    print("IMPORTANT:")
    print(
        "This is an experimental model artifact. "
        "It is not clinically validated or production-approved."
    )


if __name__ == "__main__":
    main()