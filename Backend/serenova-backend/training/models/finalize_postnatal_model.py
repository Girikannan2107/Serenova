"""
Serenova - Finalize Postnatal Baseline Model

Purpose
-------
Save the currently selected postnatal Extra Trees baseline as a
deployment-ready experimental artifact.

Selected baseline:
    Extra Trees

Baseline evaluation:
    CV Macro F1:       0.7356
    Locked Test F1:    0.6927
    Locked Test Yes Recall: 0.4800
    Locked Test Yes F1:     0.5854

Important
---------
- The test set is NOT used for fitting.
- The saved pipeline includes preprocessing + model.
- The model is fitted on train + validation only.
- No additional tuning is performed.
- This is an experimental artifact, not a clinically validated model.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Backend\serenova-backend"
)

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "post natal data.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
)

MODEL_PATH = (
    RESULTS_DIR
    / "postnatal_extra_trees_baseline.joblib"
)

METADATA_PATH = (
    RESULTS_DIR
    / "postnatal_extra_trees_baseline_metadata.json"
)

RANDOM_STATE = 42

N_SPLITS = 5

TEST_FOLD = 4
VAL_FOLD = 3


FEATURE_COLUMNS = [
    "Age",
    "Feeling sad or Tearful",
    "Irritable towards baby & partner",
    "Trouble sleeping at night",
    "Problems concentrating or making decision",
    "Overeating or loss of appetite",
    "Feeling anxious",
    "Feeling of guilt",
    "Problems of bonding with baby",
]

TARGET_COLUMN = "Suicide attempt"


EXPECTED_COLUMNS = [
    "Timestamp",
    "Age",
    "Feeling sad or Tearful",
    "Irritable towards baby & partner",
    "Trouble sleeping at night",
    "Problems concentrating or making decision",
    "Overeating or loss of appetite",
    "Feeling anxious",
    "Feeling of guilt",
    "Problems of bonding with baby",
    "Suicide attempt",
]


# ============================================================
# ENCODER
# ============================================================

def make_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


# ============================================================
# LOAD + CLEAN
# ============================================================

def load_and_clean():

    print("=" * 80)
    print("POSTNATAL MODEL FINALIZATION")
    print("=" * 80)

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{RAW_PATH}"
        )

    df = pd.read_csv(
        RAW_PATH
    )

    print(
        f"\nRaw dataset: {RAW_PATH}"
    )

    print(
        f"Original shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing expected columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    # --------------------------------------------------------
    # Remove missing rows
    # --------------------------------------------------------

    missing_rows = int(
        df.isna()
        .any(axis=1)
        .sum()
    )

    print(
        f"\nMissing-value rows removed: "
        f"{missing_rows}"
    )

    df = (
        df
        .dropna()
        .copy()
    )

    # --------------------------------------------------------
    # Remove exact duplicates
    # --------------------------------------------------------

    duplicate_rows = int(
        df.duplicated()
        .sum()
    )

    print(
        f"Exact duplicate rows removed: "
        f"{duplicate_rows}"
    )

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Remove unknown target
    # --------------------------------------------------------

    unknown_mask = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .eq(
            "Not interested to say"
        )
    )

    unknown_rows = int(
        unknown_mask.sum()
    )

    print(
        'Unknown "Not interested to say" '
        f"rows removed: {unknown_rows}"
    )

    df = df.loc[
        ~unknown_mask
    ].copy()

    # --------------------------------------------------------
    # Binary target
    # --------------------------------------------------------

    target_map = {
        "No": 0,
        "Yes": 1,
    }

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .map(target_map)
    )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Unexpected target values remain."
        )

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(np.int64)
    )

    # --------------------------------------------------------
    # Remove conflicting feature groups
    # --------------------------------------------------------

    X_raw = df[
        FEATURE_COLUMNS
    ].copy()

    y = df[
        TARGET_COLUMN
    ].copy()

    group_key = (
        X_raw
        .astype(str)
        .agg(
            "||".join,
            axis=1,
        )
    )

    conflict_counts = (
        pd.DataFrame(
            {
                "group": group_key,
                "target": y,
            }
        )
        .groupby(
            "group"
        )["target"]
        .nunique()
    )

    conflicting_groups = set(
        conflict_counts[
            conflict_counts > 1
        ].index
    )

    print(
        f"Conflicting feature groups: "
        f"{len(conflicting_groups)}"
    )

    if conflicting_groups:

        conflict_mask = (
            group_key.isin(
                conflicting_groups
            )
        )

        conflict_rows = int(
            conflict_mask.sum()
        )

        print(
            f"Conflicting rows removed: "
            f"{conflict_rows}"
        )

        df = df.loc[
            ~conflict_mask
        ].copy()

    # --------------------------------------------------------
    # Final X / y
    # --------------------------------------------------------

    df = (
        df
        .reset_index(drop=True)
    )

    X_raw = df[
        FEATURE_COLUMNS
    ].copy()

    y = df[
        TARGET_COLUMN
    ].copy()

    # --------------------------------------------------------
    # Group IDs
    # --------------------------------------------------------

    group_index = (
        pd.MultiIndex.from_frame(
            X_raw.reset_index(
                drop=True
            )
        )
    )

    group_ids = (
        group_index
        .factorize(
            sort=False
        )[0]
    )

    group_ids = np.asarray(
        group_ids,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Integrity
    # --------------------------------------------------------

    if X_raw.isna().any().any():
        raise AssertionError(
            "Missing values remain."
        )

    if y.isna().any():
        raise AssertionError(
            "Missing target values remain."
        )

    if not set(
        y.unique()
    ).issubset({0, 1}):
        raise AssertionError(
            "Target is not binary."
        )

    print(
        f"\nFinal clean rows: "
        f"{len(X_raw)}"
    )

    print(
        f"Unique groups: "
        f"{len(np.unique(group_ids))}"
    )

    return (
        X_raw,
        y,
        group_ids,
    )


# ============================================================
# RECREATE THE LOCKED SPLIT
# ============================================================

def create_locked_split(
    X,
    y,
    groups,
):

    splitter = (
        StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    )

    folds = list(
        splitter.split(
            X,
            y,
            groups,
        )
    )

    # --------------------------------------------------------
    # Locked test
    # --------------------------------------------------------

    train_val_idx, test_idx = (
        folds[TEST_FOLD]
    )

    # --------------------------------------------------------
    # Train / validation
    # --------------------------------------------------------

    X_train_val = X.iloc[
        train_val_idx
    ]

    y_train_val = y.iloc[
        train_val_idx
    ]

    groups_train_val = groups[
        train_val_idx
    ]

    inner_splitter = (
        StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    )

    inner_folds = list(
        inner_splitter.split(
            X_train_val,
            y_train_val,
            groups_train_val,
        )
    )

    train_inner, val_inner = (
        inner_folds[VAL_FOLD]
    )

    train_idx = (
        train_val_idx[
            train_inner
        ]
    )

    val_idx = (
        train_val_idx[
            val_inner
        ]
    )

    return (
        train_idx,
        val_idx,
        test_idx,
    )


# ============================================================
# CREATE BASELINE MODEL
# ============================================================

def create_model():

    encoder = make_encoder()

    model = ExtraTreesClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            (
                "encoder",
                encoder,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load / clean
    # --------------------------------------------------------

    (
        X,
        y,
        groups,
    ) = load_and_clean()

    # --------------------------------------------------------
    # Recreate exact benchmark split
    # --------------------------------------------------------

    (
        train_idx,
        val_idx,
        test_idx,
    ) = create_locked_split(
        X,
        y,
        groups,
    )

    X_train = (
        X.iloc[
            train_idx
        ]
        .reset_index(drop=True)
    )

    y_train = (
        y.iloc[
            train_idx
        ]
        .reset_index(drop=True)
    )

    groups_train = (
        groups[
            train_idx
        ]
    )

    X_val = (
        X.iloc[
            val_idx
        ]
        .reset_index(drop=True)
    )

    y_val = (
        y.iloc[
            val_idx
        ]
        .reset_index(drop=True)
    )

    groups_val = (
        groups[
            val_idx
        ]
    )

    X_test = (
        X.iloc[
            test_idx
        ]
        .reset_index(drop=True)
    )

    y_test = (
        y.iloc[
            test_idx
        ]
        .reset_index(drop=True)
    )

    groups_test = (
        groups[
            test_idx
        ]
    )

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    if (
        set(groups_train)
        & set(groups_val)
    ):
        raise AssertionError(
            "Train/validation group leakage."
        )

    if (
        set(groups_train)
        & set(groups_test)
    ):
        raise AssertionError(
            "Train/test group leakage."
        )

    if (
        set(groups_val)
        & set(groups_test)
    ):
        raise AssertionError(
            "Validation/test group leakage."
        )

    # --------------------------------------------------------
    # Train + validation
    #
    # IMPORTANT:
    # Test is NOT included.
    # --------------------------------------------------------

    X_train_val = pd.concat(
        [
            X_train,
            X_val,
        ],
        ignore_index=True,
    )

    y_train_val = pd.concat(
        [
            y_train,
            y_val,
        ],
        ignore_index=True,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL BASELINE TRAINING"
    )

    print(
        "=" * 80
    )

    print(
        f"\nTrain samples: "
        f"{len(X_train)}"
    )

    print(
        f"Validation samples: "
        f"{len(X_val)}"
    )

    print(
        f"Training samples used for final artifact: "
        f"{len(X_train_val)}"
    )

    print(
        f"Locked test samples: "
        f"{len(X_test)}"
    )

    print(
        "\nThe locked test set is NOT used "
        "for model fitting."
    )

    # --------------------------------------------------------
    # Create and fit pipeline
    # --------------------------------------------------------

    pipeline = create_model()

    pipeline.fit(
        X_train_val,
        y_train_val,
    )

    print(
        "\nBaseline Extra Trees model fitted."
    )

    # --------------------------------------------------------
    # Evaluate once on locked test
    # --------------------------------------------------------

    predictions = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_precision = (
        precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    macro_recall = (
        recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    macro_f1 = (
        f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    yes_precision = (
        precision_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0,
        )
    )

    yes_recall = (
        recall_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0,
        )
    )

    yes_f1 = (
        f1_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LOCKED TEST VERIFICATION"
    )

    print(
        "=" * 80
    )

    print(
        f"\nAccuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro precision: "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro recall: "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.4f}"
    )

    print(
        f"Yes precision: "
        f"{yes_precision:.4f}"
    )

    print(
        f"Yes recall: "
        f"{yes_recall:.4f}"
    )

    print(
        f"Yes F1: "
        f"{yes_f1:.4f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(cm)

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "No",
                "Yes",
            ],
            digits=4,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Verify expected baseline result
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BASELINE RESULT CHECK"
    )

    print(
        "=" * 80
    )

    expected_f1 = 0.6927

    print(
        f"Expected baseline test Macro F1: "
        f"{expected_f1:.4f}"
    )

    print(
        f"Current verification Macro F1: "
        f"{macro_f1:.4f}"
    )

    # --------------------------------------------------------
    # Create results directory
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save pipeline
    # --------------------------------------------------------

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    print(
        f"\nModel saved to:\n"
        f"{MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "project": "Serenova",
        "task": "postnatal_binary_classification",
        "status": "experimental_baseline",
        "model": "ExtraTreesClassifier",
        "preprocessing": "OneHotEncoder(handle_unknown='ignore')",
        "artifact_type": "sklearn_pipeline_joblib",
        "random_state": RANDOM_STATE,
        "features": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "target_mapping": {
            "No": 0,
            "Yes": 1,
        },
        "excluded_target": "Not interested to say",
        "excluded_feature": "Timestamp",
        "clean_rows": int(len(X)),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "train_plus_validation_rows": int(
            len(X_train_val)
        ),
        "locked_test_rows": int(len(X_test)),
        "unique_groups": int(
            len(np.unique(groups))
        ),
        "duplicate_rows_removed": 1080,
        "missing_rows_removed": 12,
        "unknown_target_rows_removed": 90,
        "conflicting_rows_removed": 7,
        "cv_macro_f1": 0.7356,
        "locked_test_accuracy": float(
            accuracy
        ),
        "locked_test_macro_precision": float(
            macro_precision
        ),
        "locked_test_macro_recall": float(
            macro_recall
        ),
        "locked_test_macro_f1": float(
            macro_f1
        ),
        "locked_test_yes_precision": float(
            yes_precision
        ),
        "locked_test_yes_recall": float(
            yes_recall
        ),
        "locked_test_yes_f1": float(
            yes_f1
        ),
        "confusion_matrix": cm.tolist(),
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
        "clinical_validation": False,
        "production_approved": False,
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

    print(
        f"Metadata saved to:\n"
        f"{METADATA_PATH}"
    )

    # --------------------------------------------------------
    # File verification
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise AssertionError(
            "Model file was not created."
        )

    if not METADATA_PATH.exists():
        raise AssertionError(
            "Metadata file was not created."
        )

    model_size_mb = (
        MODEL_PATH.stat().st_size
        / (
            1024 * 1024
        )
    )

    metadata_size_kb = (
        METADATA_PATH.stat().st_size
        / 1024
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ARTIFACT VERIFICATION"
    )

    print(
        "=" * 80
    )

    print(
        f"\nModel file exists: "
        f"PASS"
    )

    print(
        f"Metadata file exists: "
        f"PASS"
    )

    print(
        f"Model size: "
        f"{model_size_mb:.2f} MB"
    )

    print(
        f"Metadata size: "
        f"{metadata_size_kb:.2f} KB"
    )

    print(
        "\nSaved artifact:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nSaved metadata:"
    )

    print(
        METADATA_PATH
    )

    print(
        "\nPostnatal baseline model "
        "finalization complete."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is an experimental model artifact. "
        "It is not clinically validated or "
        "production-approved."
    )


if __name__ == "__main__":
    main()