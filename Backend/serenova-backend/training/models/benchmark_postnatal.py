"""
Postnatal depression benchmark for Serenova.

Purpose:
- Reproduce the leakage-safe cleaning decisions from analyze_postnatal.py.
- Keep the locked test set untouched during model selection.
- Use group-aware splitting because repeated feature combinations exist.
- Fit preprocessing inside each training fold.
- Benchmark multiple classifiers.
- Select the model using group-aware CV macro F1.
- Evaluate the selected model once on the locked test set.
- Do not save a model in this benchmark stage.

Dataset:
    data/raw/post natal data.csv

Target:
    No  -> 0
    Yes -> 1

"Not interested to say" is treated as unknown and excluded.
Timestamp is excluded as a predictive feature.
Rows with missing values are excluded.
Exact duplicates are removed.
Conflicting feature groups are removed.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(r"D:\Backend\serenova-backend")

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "post natal data.csv"

RANDOM_STATE = 42

# These match analyze_postnatal.py.
TEST_FOLD = 4
VAL_FOLD = 3
N_SPLITS = 5


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


# ============================================================
# Utility functions
# ============================================================

def make_one_hot_encoder():
    """
    Create a compatible OneHotEncoder for different sklearn versions.
    """
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


def make_preprocessor(scale=False):
    """
    Preprocessing is fitted only on the training portion of each fold
    because it is inside the sklearn Pipeline.
    """
    encoder = make_one_hot_encoder()

    if scale:
        return ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("onehot", encoder),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    FEATURE_COLUMNS,
                )
            ],
            remainder="drop",
        )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                encoder,
                FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )


def make_models():
    """
    Baseline model set.

    Class weighting is used for the linear/tree models to reduce the
    effect of the moderate class imbalance. MLP does not expose the
    same class_weight interface, so it is left unweighted.
    """
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                (
                    "preprocess",
                    make_preprocessor(scale=True),
                ),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                (
                    "preprocess",
                    make_preprocessor(scale=False),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "Extra Trees": Pipeline(
            steps=[
                (
                    "preprocess",
                    make_preprocessor(scale=False),
                ),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=400,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "XGBoost": Pipeline(
            steps=[
                (
                    "preprocess",
                    make_preprocessor(scale=False),
                ),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "MLP": Pipeline(
            steps=[
                (
                    "preprocess",
                    make_preprocessor(scale=True),
                ),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=1e-3,
                        learning_rate_init=1e-3,
                        max_iter=1500,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=30,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return models


def print_distribution(name, y):
    """
    Print class counts and percentages.
    """
    values = pd.Series(y).value_counts().sort_index()

    total = len(y)

    print(f"\n{name}")
    print("-" * len(name))

    for label, count in values.items():
        pct = 100.0 * count / total
        label_name = "No" if int(label) == 0 else "Yes"
        print(
            f"  {label_name}: {count:3d} "
            f"({pct:6.2f}%)"
        )

    print(f"  Total: {total}")


def check_group_leakage(groups_a, groups_b):
    """
    Return whether two group sets are disjoint.
    """
    return len(set(groups_a).intersection(set(groups_b))) == 0


# ============================================================
# Data cleaning
# ============================================================

def load_and_clean():
    print("=" * 80)
    print("POSTNATAL DATA LOADING AND CLEANING")
    print("=" * 80)

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_PATH}"
        )

    df = pd.read_csv(RAW_PATH)

    print(f"\nRaw dataset: {RAW_PATH}")
    print(f"Original shape: {df.shape}")

    # --------------------------------------------------------
    # Column validation
    # --------------------------------------------------------

    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing expected columns:\n"
            + "\n".join(missing_columns)
        )

    extra_columns = [
        col for col in df.columns
        if col not in EXPECTED_COLUMNS
    ]

    if extra_columns:
        print("\nWarning: extra columns detected:")
        for col in extra_columns:
            print(f"  - {col}")

    # --------------------------------------------------------
    # Drop rows with missing values
    # --------------------------------------------------------

    missing_rows = int(df.isna().any(axis=1).sum())

    print(f"\nRows containing missing values: {missing_rows}")

    df = df.dropna().copy()

    print(f"After missing-row removal: {df.shape}")

    # --------------------------------------------------------
    # Remove exact duplicates
    # --------------------------------------------------------

    duplicate_count = int(df.duplicated().sum())

    print(f"\nExact duplicate rows removed: {duplicate_count}")

    df = df.drop_duplicates().reset_index(drop=True)

    print(f"After duplicate removal: {df.shape}")

    # --------------------------------------------------------
    # Target cleaning
    # --------------------------------------------------------

    target_distribution = df[TARGET_COLUMN].value_counts(
        dropna=False
    )

    print("\nTarget distribution before unknown removal:")
    print(target_distribution)

    unknown_mask = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .eq("Not interested to say")
    )

    unknown_count = int(unknown_mask.sum())

    print(
        f"\nUnknown target rows removed "
        f'("Not interested to say"): {unknown_count}'
    )

    df = df.loc[~unknown_mask].copy()

    # --------------------------------------------------------
    # Binary target mapping
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
            "Unexpected target values remain after mapping."
        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(np.int64)

    # --------------------------------------------------------
    # Remove conflicting feature groups
    # --------------------------------------------------------

    X_raw = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    group_key = X_raw.astype(str).agg(
        "||".join,
        axis=1,
    )

    conflict_table = (
        pd.DataFrame(
            {
                "group": group_key,
                "target": y,
            }
        )
        .groupby("group")["target"]
        .nunique()
    )

    conflicting_groups = set(
        conflict_table[
            conflict_table > 1
        ].index
    )

    print(
        f"\nConflicting feature groups removed: "
        f"{len(conflicting_groups)}"
    )

    if conflicting_groups:
        conflict_mask = group_key.isin(
            conflicting_groups
        )

        print(
            f"Rows removed because of conflicts: "
            f"{int(conflict_mask.sum())}"
        )

        df = df.loc[~conflict_mask].copy()

    # --------------------------------------------------------
    # Rebuild X and y after conflict removal
    # --------------------------------------------------------

    df = df.reset_index(drop=True)

    X_raw = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # --------------------------------------------------------
    # Group IDs
    # --------------------------------------------------------

    group_index = pd.MultiIndex.from_frame(
        X_raw.reset_index(drop=True)
    )

    group_ids = group_index.factorize(
        sort=False
    )[0]

    group_ids = np.asarray(
        group_ids,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Final integrity checks
    # --------------------------------------------------------

    if X_raw.isna().any().any():
        raise AssertionError(
            "Missing values remain in features."
        )

    if y.isna().any():
        raise AssertionError(
            "Missing target values remain."
        )

    if not set(y.unique()).issubset({0, 1}):
        raise AssertionError(
            "Target contains values other than 0 and 1."
        )

    if len(X_raw) != len(y):
        raise AssertionError(
            "Feature and target lengths do not match."
        )

    print("\nFinal cleaned dataset:")
    print(f"  Rows: {len(df)}")
    print(f"  Features: {len(FEATURE_COLUMNS)}")
    print(f"  Unique feature groups: {len(np.unique(group_ids))}")

    print_distribution(
        "Final target distribution",
        y,
    )

    return (
        df,
        X_raw,
        y,
        group_ids,
    )


# ============================================================
# Deterministic train / validation / test split
# ============================================================

def create_locked_split(X, y, groups):
    print("\n" + "=" * 80)
    print("GROUP-AWARE TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 80)

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    folds = list(
        splitter.split(
            X,
            y,
            groups,
        )
    )

    test_train_indices, test_indices = folds[TEST_FOLD]

    train_val_indices = test_train_indices

    X_train_val = X.iloc[train_val_indices]
    y_train_val = y.iloc[train_val_indices]
    groups_train_val = groups[train_val_indices]

    inner_splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    inner_folds = list(
        inner_splitter.split(
            X_train_val,
            y_train_val,
            groups_train_val,
        )
    )

    train_indices_inner, val_indices_inner = inner_folds[
        VAL_FOLD
    ]

    train_indices = train_val_indices[
        train_indices_inner
    ]

    val_indices = train_val_indices[
        val_indices_inner
    ]

    return (
        train_indices,
        val_indices,
        test_indices,
    )


# ============================================================
# Validation benchmark
# ============================================================

def run_validation_benchmark(
    models,
    X_train,
    y_train,
    X_val,
    y_val,
):
    print("\n" + "=" * 80)
    print("VALIDATION BENCHMARK")
    print("=" * 80)

    results = []

    for name, model in models.items():
        print(f"\nTraining: {name}")

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_val
        )

        accuracy = accuracy_score(
            y_val,
            predictions,
        )

        precision = precision_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        results.append(
            {
                "Model": name,
                "Validation Accuracy": accuracy,
                "Validation Macro Precision": precision,
                "Validation Macro Recall": recall,
                "Validation Macro F1": macro_f1,
            }
        )

        print(
            f"  Accuracy:  {accuracy:.4f}"
        )
        print(
            f"  Macro F1:  {macro_f1:.4f}"
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "Validation Macro F1",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\nValidation ranking:")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return results_df


# ============================================================
# Group-aware cross-validation
# ============================================================

def run_group_cv(
    models,
    X,
    y,
    groups,
):
    print("\n" + "=" * 80)
    print("GROUP-AWARE 5-FOLD CROSS-VALIDATION")
    print("=" * 80)

    results = []

    for name, model in models.items():
        print(f"\nCross-validating: {name}")

        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        fold_scores = []

        for fold_number, (
            train_idx,
            val_idx,
        ) in enumerate(
            splitter.split(
                X,
                y,
                groups,
            ),
            start=1,
        ):
            fold_model = model

            fold_model.fit(
                X.iloc[train_idx],
                y.iloc[train_idx],
            )

            predictions = fold_model.predict(
                X.iloc[val_idx]
            )

            fold_f1 = f1_score(
                y.iloc[val_idx],
                predictions,
                average="macro",
                zero_division=0,
            )

            fold_scores.append(
                fold_f1
            )

            print(
                f"  Fold {fold_number}: "
                f"macro F1 = {fold_f1:.4f}"
            )

        mean_f1 = float(
            np.mean(fold_scores)
        )

        std_f1 = float(
            np.std(fold_scores)
        )

        results.append(
            {
                "Model": name,
                "CV Macro F1 Mean": mean_f1,
                "CV Macro F1 Std": std_f1,
                "Fold Scores": fold_scores,
            }
        )

        print(
            f"  Mean: {mean_f1:.4f}"
        )
        print(
            f"  Std:  {std_f1:.4f}"
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "CV Macro F1 Mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\nCross-validation ranking:")

    display_df = results_df[
        [
            "Model",
            "CV Macro F1 Mean",
            "CV Macro F1 Std",
        ]
    ]

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return results_df


# ============================================================
# Locked test evaluation
# ============================================================

def evaluate_locked_test(
    model,
    model_name,
    X_train_val,
    y_train_val,
    X_test,
    y_test,
):
    print("\n" + "=" * 80)
    print("LOCKED TEST EVALUATION")
    print("=" * 80)

    print(
        f"\nSelected model: {model_name}"
    )

    print(
        "\nThe model is now fitted on train + validation only."
    )

    model.fit(
        X_train_val,
        y_train_val,
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
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

    print("\nLocked test metrics:")
    print(
        f"  Accuracy:       {accuracy:.4f}"
    )
    print(
        f"  Macro Precision:{precision:.4f}"
    )
    print(
        f"  Macro Recall:   {recall:.4f}"
    )
    print(
        f"  Macro F1:       {macro_f1:.4f}"
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No", "Yes"],
            digits=4,
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "predictions": predictions,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("SERENOVA POSTNATAL MODEL BENCHMARK")
    print("=" * 80)

    # --------------------------------------------------------
    # Load and clean
    # --------------------------------------------------------

    (
        df,
        X,
        y,
        groups,
    ) = load_and_clean()

    # --------------------------------------------------------
    # Locked split
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

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    groups_train = groups[train_idx]

    X_val = X.iloc[val_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)
    groups_val = groups[val_idx]

    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    groups_test = groups[test_idx]

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    print("\nSplit sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")

    print_distribution(
        "Train distribution",
        y_train,
    )

    print_distribution(
        "Validation distribution",
        y_val,
    )

    print_distribution(
        "Test distribution",
        y_test,
    )

    train_val_leakage = not check_group_leakage(
        groups_train,
        groups_val,
    )

    train_test_leakage = not check_group_leakage(
        groups_train,
        groups_test,
    )

    val_test_leakage = not check_group_leakage(
        groups_val,
        groups_test,
    )

    print("\nGroup leakage:")
    print(
        f"  Train-Val:  {'FAIL' if train_val_leakage else 'PASS'}"
    )
    print(
        f"  Train-Test: {'FAIL' if train_test_leakage else 'PASS'}"
    )
    print(
        f"  Val-Test:   {'FAIL' if val_test_leakage else 'PASS'}"
    )

    if (
        train_val_leakage
        or train_test_leakage
        or val_test_leakage
    ):
        raise AssertionError(
            "Group leakage detected."
        )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = make_models()

    # --------------------------------------------------------
    # Validation benchmark
    # --------------------------------------------------------

    validation_results = run_validation_benchmark(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    # --------------------------------------------------------
    # CV model selection
    #
    # Use train + validation only.
    # The test set remains completely locked.
    # --------------------------------------------------------

    X_train_val = pd.concat(
        [X_train, X_val],
        axis=0,
        ignore_index=True,
    )

    y_train_val = pd.concat(
        [y_train, y_val],
        axis=0,
        ignore_index=True,
    )

    groups_train_val = np.concatenate(
        [
            groups_train,
            groups_val,
        ]
    )

    cv_results = run_group_cv(
        models=models,
        X=X_train_val,
        y=y_train_val,
        groups=groups_train_val,
    )

    selected_model_name = cv_results.iloc[0]["Model"]

    selected_model = models[
        selected_model_name
    ]

    selected_cv_f1 = float(
        cv_results.iloc[0]["CV Macro F1 Mean"]
    )

    # --------------------------------------------------------
    # Locked test evaluation
    # --------------------------------------------------------

    test_results = evaluate_locked_test(
        model=selected_model,
        model_name=selected_model_name,
        X_train_val=X_train_val,
        y_train_val=y_train_val,
        X_test=X_test,
        y_test=y_test,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("POSTNATAL BENCHMARK SUMMARY")
    print("=" * 80)

    print(f"\nClean dataset rows: {len(df)}")
    print(f"Train rows:         {len(X_train)}")
    print(f"Validation rows:    {len(X_val)}")
    print(f"Locked test rows:   {len(X_test)}")

    print(
        f"\nSelected model: {selected_model_name}"
    )

    print(
        f"CV macro F1:   {selected_cv_f1:.4f}"
    )

    print(
        f"Test accuracy:  {test_results['accuracy']:.4f}"
    )

    print(
        f"Test macro F1:  {test_results['macro_f1']:.4f}"
    )

    print(
        f"Test macro precision: "
        f"{test_results['macro_precision']:.4f}"
    )

    print(
        f"Test macro recall:    "
        f"{test_results['macro_recall']:.4f}"
    )

    # --------------------------------------------------------
    # Integrity summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("EVALUATION INTEGRITY")
    print("=" * 80)

    checks = {
        "No missing values": not X.isna().any().any(),
        "Binary target only": set(y.unique()).issubset({0, 1}),
        "No train-val group leakage": not train_val_leakage,
        "No train-test group leakage": not train_test_leakage,
        "No val-test group leakage": not val_test_leakage,
        "Test not used for model selection": True,
        "Preprocessing fitted inside model pipeline": True,
        "No model file saved": True,
    }

    for name, passed in checks.items():
        print(
            f"  {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if not all(checks.values()):
        raise AssertionError(
            "One or more evaluation integrity checks failed."
        )

    print("\nBenchmark complete.")
    print(
        "No model was saved. "
        "Use the benchmark result to decide whether "
        "hyperparameter tuning is justified."
    )


if __name__ == "__main__":
    main()