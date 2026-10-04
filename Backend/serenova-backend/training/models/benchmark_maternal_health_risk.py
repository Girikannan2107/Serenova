"""
Serenova - Maternal Health Risk
Class-Balanced Group-Aware Benchmark

Two evaluations are performed:

A. FULL DATASET
   - Exact duplicate rows removed
   - Conflicting feature/label groups retained
   - Group-aware split
   - Class-distribution-aware split

B. CLEAN DATASET
   - Exact duplicate rows removed
   - Feature groups with conflicting labels removed
   - Group-aware split
   - Class-distribution-aware split

No final model is saved by this script.

The test set is locked and used only for final evaluation.
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_val_score,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parents[2]

CSV_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "Maternal Health Risk Data Set.csv"
)

FEATURE_COLUMNS = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
]

TARGET_COLUMN = "RiskLevel"

LABEL_MAPPING = {
    "low risk": 0,
    "mid risk": 1,
    "high risk": 2,
}

REVERSE_LABEL_MAPPING = {
    0: "low risk",
    1: "mid risk",
    2: "high risk",
}


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 80)
print("SERENOVA - MATERNAL HEALTH RISK")
print("CLASS-BALANCED GROUP-AWARE BENCHMARK")
print("=" * 80)

print(f"\nProject directory:")
print(BASE_DIR)

print(f"\nCSV:")
print(CSV_PATH)


# ============================================================
# LOAD DATA
# ============================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"\nCSV file not found:\n{CSV_PATH}"
    )


print("\n" + "=" * 80)
print("1. LOADING DATA")
print("=" * 80)

df = pd.read_csv(CSV_PATH)

print(
    f"\nOriginal dataset shape: "
    f"{df.shape}"
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = (
    FEATURE_COLUMNS
    + [TARGET_COLUMN]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

print("\nRequired columns found.")


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing values:")

print(
    df[required_columns]
    .isnull()
    .sum()
)

if (
    df[required_columns]
    .isnull()
    .sum()
    .sum()
    > 0
):
    raise ValueError(
        "Dataset contains missing values."
    )

print("\nNo missing values.")


# ============================================================
# NORMALIZE TARGET
# ============================================================

df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .astype(str)
    .str.strip()
    .str.lower()
)


unknown_labels = (
    set(df[TARGET_COLUMN].unique())
    - set(LABEL_MAPPING.keys())
)

if unknown_labels:
    raise ValueError(
        f"Unknown labels: {unknown_labels}"
    )


# ============================================================
# NUMERIC FEATURES
# ============================================================

for column in FEATURE_COLUMNS:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


if (
    df[FEATURE_COLUMNS]
    .isnull()
    .sum()
    .sum()
    > 0
):
    raise ValueError(
        "Non-numeric feature values detected."
    )


# ============================================================
# FINITE CHECK
# ============================================================

X_original = df[
    FEATURE_COLUMNS
].to_numpy(
    dtype=np.float32
)

if not np.isfinite(
    X_original
).all():

    raise ValueError(
        "NaN or infinite values detected."
    )

print(
    "\nAll feature values are finite."
)


# ============================================================
# ORIGINAL CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("2. ORIGINAL DATASET")
print("=" * 80)

print(
    f"\nOriginal rows: "
    f"{len(df)}"
)

print("\nOriginal class distribution:")

original_counts = (
    df[TARGET_COLUMN]
    .value_counts()
)

for label in [
    "low risk",
    "mid risk",
    "high risk",
]:

    count = int(
        original_counts.get(
            label,
            0
        )
    )

    percentage = (
        count
        / len(df)
        * 100
    )

    print(
        f"  {label:<10} "
        f"{count:>5} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# EXACT DUPLICATES
# ============================================================

print("\n" + "=" * 80)
print("3. EXACT DUPLICATE ANALYSIS")
print("=" * 80)

rows_before = len(df)

exact_duplicate_rows = int(
    df.duplicated(
        keep=False
    ).sum()
)

exact_duplicates_removed = int(
    df.duplicated(
        keep="first"
    ).sum()
)

print(
    f"\nRows participating in exact duplicates: "
    f"{exact_duplicate_rows}"
)

print(
    f"Exact duplicate rows removed: "
    f"{exact_duplicates_removed}"
)

df = (
    df
    .drop_duplicates(
        keep="first"
    )
    .reset_index(
        drop=True
    )
)

print(
    f"\nRows after exact duplicate removal: "
    f"{len(df)}"
)


# ============================================================
# CREATE FEATURE GROUP
# ============================================================

print("\n" + "=" * 80)
print("4. FEATURE GROUP ANALYSIS")
print("=" * 80)

df["_feature_group"] = (
    df[FEATURE_COLUMNS]
    .astype(str)
    .agg(
        "|".join,
        axis=1
    )
)

group_sizes = (
    df.groupby(
        "_feature_group"
    )
    .size()
)

repeated_groups = int(
    (group_sizes > 1)
    .sum()
)

print(
    f"\nUnique feature combinations: "
    f"{len(group_sizes)}"
)

print(
    f"Repeated feature combinations: "
    f"{repeated_groups}"
)

print(
    f"Largest feature group: "
    f"{group_sizes.max()}"
)


# ============================================================
# CONFLICTING GROUPS
# ============================================================

labels_per_group = (
    df.groupby(
        "_feature_group"
    )[TARGET_COLUMN]
    .nunique()
)

conflicting_group_ids = set(
    labels_per_group[
        labels_per_group > 1
    ].index
)

print(
    f"\nConflicting feature groups: "
    f"{len(conflicting_group_ids)}"
)


# ============================================================
# SHOW CONFLICT EXAMPLES
# ============================================================

if conflicting_group_ids:

    print(
        "\nExample conflicting groups:"
    )

    shown = 0

    for group_id in conflicting_group_ids:

        rows = df[
            df["_feature_group"]
            == group_id
        ]

        print("\n----------------------------------------")

        print(
            rows[
                FEATURE_COLUMNS
                + [TARGET_COLUMN]
            ].to_string(
                index=False
            )
        )

        shown += 1

        if shown >= 5:
            break


# ============================================================
# PREPARE FULL DATASET
# ============================================================

full_df = df.copy()


# ============================================================
# PREPARE CLEAN DATASET
# ============================================================

clean_df = df[
    ~df["_feature_group"].isin(
        conflicting_group_ids
    )
].copy()

print("\n" + "=" * 80)
print("5. FULL VS CLEAN DATASET")
print("=" * 80)

print(
    f"\nFULL dataset rows: "
    f"{len(full_df)}"
)

print(
    f"CLEAN dataset rows: "
    f"{len(clean_df)}"
)

print(
    f"Rows excluded from CLEAN dataset: "
    f"{len(full_df) - len(clean_df)}"
)

print(
    "\nCLEAN dataset excludes only feature "
    "combinations with conflicting labels."
)


# ============================================================
# DATASET PREPARATION FUNCTION
# ============================================================

def prepare_arrays(dataset):

    X = dataset[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    y = dataset[
        TARGET_COLUMN
    ].map(
        LABEL_MAPPING
    ).to_numpy(
        dtype=np.int64
    )

    groups = dataset[
        "_feature_group"
    ].to_numpy()

    return X, y, groups


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def get_distribution(
    y
):

    counts = np.bincount(
        y,
        minlength=3
    )

    proportions = (
        counts
        / len(y)
    )

    return proportions


# ============================================================
# CLASS-BALANCED GROUP SPLIT
# ============================================================

def choose_group_split(
    X,
    y,
    groups,
    desired_fraction,
    random_state,
    n_splits=5,
):
    """
    Select a StratifiedGroupKFold fold whose:

    1. Evaluation size is close to desired_fraction.
    2. Class distribution is close to the
       overall class distribution.

    Groups are never split.
    """

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    overall_distribution = (
        get_distribution(y)
    )

    total_samples = len(y)

    candidates = []

    for fold_number, (
        train_idx,
        eval_idx,
    ) in enumerate(
        splitter.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):

        eval_fraction = (
            len(eval_idx)
            / total_samples
        )

        eval_distribution = (
            get_distribution(
                y[eval_idx]
            )
        )

        # L1 class-distribution error.
        class_error = np.sum(
            np.abs(
                eval_distribution
                - overall_distribution
            )
        )

        size_error = abs(
            eval_fraction
            - desired_fraction
        )

        # Class balance is weighted more heavily.
        score = (
            class_error * 10.0
            + size_error
        )

        candidates.append(
            (
                score,
                fold_number,
                train_idx,
                eval_idx,
                eval_fraction,
                class_error,
            )
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    return candidates[0]


# ============================================================
# CREATE TRAIN / VAL / TEST
# ============================================================

def make_dataset_split(
    dataset_name,
    X,
    y,
    groups,
):

    print("\n" + "-" * 80)
    print(
        f"CREATING SPLIT: "
        f"{dataset_name}"
    )
    print("-" * 80)

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    (
        test_score,
        test_fold,
        train_val_idx,
        test_idx,
        test_fraction,
        test_class_error,
    ) = choose_group_split(
        X,
        y,
        groups,
        desired_fraction=0.20,
        random_state=RANDOM_STATE,
    )

    X_train_val = X[
        train_val_idx
    ]

    y_train_val = y[
        train_val_idx
    ]

    groups_train_val = groups[
        train_val_idx
    ]

    X_test = X[
        test_idx
    ]

    y_test = y[
        test_idx
    ]

    groups_test = groups[
        test_idx
    ]

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    (
        val_score,
        val_fold,
        train_idx,
        val_idx,
        val_fraction,
        val_class_error,
    ) = choose_group_split(
        X_train_val,
        y_train_val,
        groups_train_val,
        desired_fraction=0.25,
        random_state=RANDOM_STATE + 1,
    )

    X_train = X_train_val[
        train_idx
    ]

    y_train = y_train_val[
        train_idx
    ]

    groups_train = groups_train_val[
        train_idx
    ]

    X_val = X_train_val[
        val_idx
    ]

    y_val = y_train_val[
        val_idx
    ]

    groups_val = groups_train_val[
        val_idx
    ]

    print(
        f"\nTest fold: "
        f"{test_fold}"
    )

    print(
        f"Validation fold: "
        f"{val_fold}"
    )

    print(
        f"\nTrain: "
        f"{len(X_train)}"
    )

    print(
        f"Validation: "
        f"{len(X_val)}"
    )

    print(
        f"Test: "
        f"{len(X_test)}"
    )

    # --------------------------------------------------------
    # LEAKAGE CHECK
    # --------------------------------------------------------

    train_groups = set(
        groups_train
    )

    val_groups = set(
        groups_val
    )

    test_groups = set(
        groups_test
    )

    train_val_overlap = (
        train_groups
        & val_groups
    )

    train_test_overlap = (
        train_groups
        & test_groups
    )

    val_test_overlap = (
        val_groups
        & test_groups
    )

    print(
        "\nGroup leakage:"
    )

    print(
        f"  Train ∩ Validation: "
        f"{len(train_val_overlap)}"
    )

    print(
        f"  Train ∩ Test: "
        f"{len(train_test_overlap)}"
    )

    print(
        f"  Validation ∩ Test: "
        f"{len(val_test_overlap)}"
    )

    if (
        train_val_overlap
        or train_test_overlap
        or val_test_overlap
    ):

        raise RuntimeError(
            "GROUP LEAKAGE DETECTED."
        )

    print(
        "\nPASS: No group leakage."
    )

    # --------------------------------------------------------
    # DISTRIBUTIONS
    # --------------------------------------------------------

    def print_dist(
        name,
        labels,
    ):

        distribution = (
            get_distribution(
                labels
            )
        )

        print(
            f"\n{name}:"
        )

        for class_id in range(3):

            count = int(
                np.sum(
                    labels
                    == class_id
                )
            )

            percentage = (
                distribution[class_id]
                * 100
            )

            print(
                f"  "
                f"{REVERSE_LABEL_MAPPING[class_id]:<10}"
                f"{count:>5} "
                f"({percentage:.2f}%)"
            )

    print_dist(
        "TRAIN",
        y_train,
    )

    print_dist(
        "VALIDATION",
        y_val,
    )

    print_dist(
        "TEST",
        y_test,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "groups_train": groups_train,

        "X_val": X_val,
        "y_val": y_val,
        "groups_val": groups_val,

        "X_train_val": X_train_val,
        "y_train_val": y_train_val,
        "groups_train_val": groups_train_val,

        "X_test": X_test,
        "y_test": y_test,
        "groups_test": groups_test,
    }


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(
    name
):

    if name == "RandomForest":

        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if name == "ExtraTrees":

        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if name == "XGBoost":

        return XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            min_child_weight=1,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            device="cuda",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if name == "MLP":

        return Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(
                            128,
                            64,
                            32,
                        ),
                        activation="relu",
                        solver="adam",
                        alpha=0.0001,
                        batch_size=32,
                        learning_rate_init=0.001,
                        max_iter=1000,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=30,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    raise ValueError(
        f"Unknown model: {name}"
    )


# ============================================================
# MODEL BENCHMARK FUNCTION
# ============================================================

def benchmark_dataset(
    dataset_name,
    split_data,
):

    print("\n\n" + "=" * 80)
    print(
        f"BENCHMARK: {dataset_name}"
    )
    print("=" * 80)

    X_train = split_data[
        "X_train"
    ]

    y_train = split_data[
        "y_train"
    ]

    X_val = split_data[
        "X_val"
    ]

    y_val = split_data[
        "y_val"
    ]

    X_train_val = split_data[
        "X_train_val"
    ]

    y_train_val = split_data[
        "y_train_val"
    ]

    groups_train_val = split_data[
        "groups_train_val"
    ]

    X_test = split_data[
        "X_test"
    ]

    y_test = split_data[
        "y_test"
    ]

    model_names = [
        "RandomForest",
        "ExtraTrees",
        "XGBoost",
        "MLP",
    ]

    validation_results = []

    # ========================================================
    # VALIDATION BENCHMARK
    # ========================================================

    print("\n" + "-" * 80)
    print("VALIDATION BENCHMARK")
    print("-" * 80)

    for model_name in model_names:

        print(
            f"\nTraining: "
            f"{model_name}"
        )

        model = create_model(
            model_name
        )

        start_time = time.time()

        model.fit(
            X_train,
            y_train,
        )

        elapsed = (
            time.time()
            - start_time
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

        print(
            f"  Accuracy : {accuracy:.4f}"
        )

        print(
            f"  Macro F1 : {macro_f1:.4f}"
        )

        print(
            f"  Time     : {elapsed:.2f}s"
        )

        validation_results.append(
            {
                "name": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "macro_f1": macro_f1,
            }
        )

    # ========================================================
    # GROUP-AWARE CV FOR EVERY MODEL
    # ========================================================

    print("\n" + "-" * 80)
    print(
        "GROUP-AWARE 5-FOLD CROSS-VALIDATION"
    )
    print("-" * 80)

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_results = []

    for model_name in model_names:

        print(
            f"\nCV: {model_name}"
        )

        model = create_model(
            model_name
        )

        accuracy_scores = cross_val_score(
            model,
            X_train_val,
            y_train_val,
            groups=groups_train_val,
            cv=cv,
            scoring="accuracy",
            n_jobs=None,
        )

        f1_model = create_model(
            model_name
        )

        f1_scores = cross_val_score(
            f1_model,
            X_train_val,
            y_train_val,
            groups=groups_train_val,
            cv=cv,
            scoring="f1_macro",
            n_jobs=None,
        )

        print(
            "  Accuracy:"
        )

        for index, score in enumerate(
            accuracy_scores,
            start=1,
        ):

            print(
                f"    Fold {index}: "
                f"{score:.4f}"
            )

        print(
            f"  Mean accuracy: "
            f"{accuracy_scores.mean():.4f}"
        )

        print(
            f"  Std accuracy: "
            f"{accuracy_scores.std():.4f}"
        )

        print(
            f"  Mean macro F1: "
            f"{f1_scores.mean():.4f}"
        )

        print(
            f"  Std macro F1: "
            f"{f1_scores.std():.4f}"
        )

        cv_results.append(
            {
                "name": model_name,
                "cv_accuracy": accuracy_scores.mean(),
                "cv_accuracy_std": accuracy_scores.std(),
                "cv_macro_f1": f1_scores.mean(),
                "cv_macro_f1_std": f1_scores.std(),
            }
        )

    # ========================================================
    # COMBINED RESULTS
    # ========================================================

    print("\n" + "=" * 80)
    print(
        f"{dataset_name} - MODEL COMPARISON"
    )
    print("=" * 80)

    comparison = []

    for validation in validation_results:

        cv_result = next(
            item
            for item in cv_results
            if item["name"]
            == validation["name"]
        )

        comparison.append(
            {
                **validation,
                **cv_result,
            }
        )

    # Use CV macro-F1 as the primary selection metric.
    # Accuracy is secondary.
    comparison.sort(
        key=lambda item: (
            item["cv_macro_f1"],
            item["cv_accuracy"],
            item["accuracy"],
        ),
        reverse=True,
    )

    print(
        "\n"
        f"{'Model':<18}"
        f"{'Val Acc':>10}"
        f"{'Val F1':>10}"
        f"{'CV Acc':>10}"
        f"{'CV F1':>10}"
    )

    print(
        "-" * 58
    )

    for item in comparison:

        print(
            f"{item['name']:<18}"
            f"{item['accuracy']:>10.4f}"
            f"{item['macro_f1']:>10.4f}"
            f"{item['cv_accuracy']:>10.4f}"
            f"{item['cv_macro_f1']:>10.4f}"
        )

    selected_name = comparison[0][
        "name"
    ]

    print(
        f"\nSelected model: "
        f"{selected_name}"
    )

    print(
        "Selection basis: "
        "group-aware CV macro-F1, "
        "then CV accuracy."
    )

    # ========================================================
    # FINAL TRAINING
    # ========================================================

    print("\n" + "-" * 80)
    print("FINAL TRAINING")
    print("-" * 80)

    final_model = create_model(
        selected_name
    )

    start_time = time.time()

    final_model.fit(
        X_train_val,
        y_train_val,
    )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"\nTraining time: "
        f"{elapsed:.2f}s"
    )

    # ========================================================
    # LOCKED TEST
    # ========================================================

    print("\n" + "-" * 80)
    print("LOCKED TEST EVALUATION")
    print("-" * 80)

    test_predictions = (
        final_model.predict(
            X_test
        )
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

    test_macro_f1 = f1_score(
        y_test,
        test_predictions,
        average="macro",
        zero_division=0,
    )

    print(
        f"\nFinal model: "
        f"{selected_name}"
    )

    print(
        f"Test Accuracy : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test Precision: "
        f"{test_precision:.4f}"
    )

    print(
        f"Test Recall   : "
        f"{test_recall:.4f}"
    )

    print(
        f"Test Macro F1 : "
        f"{test_macro_f1:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            test_predictions,
        )
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            test_predictions,
            labels=[0, 1, 2],
            target_names=[
                "low risk",
                "mid risk",
                "high risk",
            ],
            digits=4,
            zero_division=0,
        )
    )

    return {
        "dataset": dataset_name,
        "selected_model": selected_name,
        "comparison": comparison,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_macro_f1": test_macro_f1,
        "test_size": len(y_test),
    }


# ============================================================
# PREPARE FULL DATASET
# ============================================================

X_full, y_full, groups_full = (
    prepare_arrays(
        full_df
    )
)


# ============================================================
# PREPARE CLEAN DATASET
# ============================================================

X_clean, y_clean, groups_clean = (
    prepare_arrays(
        clean_df
    )
)


# ============================================================
# FULL SPLIT
# ============================================================

full_split = make_dataset_split(
    "FULL DATASET",
    X_full,
    y_full,
    groups_full,
)


# ============================================================
# CLEAN SPLIT
# ============================================================

clean_split = make_dataset_split(
    "CLEAN DATASET",
    X_clean,
    y_clean,
    groups_clean,
)


# ============================================================
# RUN FULL BENCHMARK
# ============================================================

full_result = benchmark_dataset(
    "FULL DATASET",
    full_split,
)


# ============================================================
# RUN CLEAN BENCHMARK
# ============================================================

clean_result = benchmark_dataset(
    "CLEAN DATASET",
    clean_split,
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n" + "=" * 80)
print("FINAL FULL VS CLEAN COMPARISON")
print("=" * 80)

print(
    "\n"
    f"{'Dataset':<18}"
    f"{'Rows':>10}"
    f"{'Model':>18}"
    f"{'Test Acc':>12}"
    f"{'Test F1':>12}"
)

print(
    "-" * 70
)

print(
    f"{'FULL':<18}"
    f"{len(full_df):>10}"
    f"{full_result['selected_model']:>18}"
    f"{full_result['test_accuracy']:>12.4f}"
    f"{full_result['test_macro_f1']:>12.4f}"
)

print(
    f"{'CLEAN':<18}"
    f"{len(clean_df):>10}"
    f"{clean_result['selected_model']:>18}"
    f"{clean_result['test_accuracy']:>12.4f}"
    f"{clean_result['test_macro_f1']:>12.4f}"
)


# ============================================================
# DATA QUALITY SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("DATA QUALITY SUMMARY")
print("=" * 80)

print(
    f"""
Original rows:
    {rows_before}

Exact duplicate rows removed:
    {exact_duplicates_removed}

Rows after exact duplicate removal:
    {len(full_df)}

Unique feature combinations:
    {len(group_sizes)}

Repeated feature combinations:
    {repeated_groups}

Conflicting feature combinations:
    {len(conflicting_group_ids)}

FULL dataset:
    {len(full_df)} rows

CLEAN dataset:
    {len(clean_df)} rows

Rows excluded from CLEAN:
    {len(full_df) - len(clean_df)}
"""
)


# ============================================================
# INTEGRITY
# ============================================================

print("=" * 80)
print("EVALUATION INTEGRITY")
print("=" * 80)

print(
    "\nPASS - exact duplicate analysis"
)

print(
    "PASS - conflicting-label analysis"
)

print(
    "PASS - group-aware splitting"
)

print(
    "PASS - class-distribution-aware split"
)

print(
    "PASS - no group leakage"
)

print(
    "PASS - locked test evaluation"
)

print(
    "\nNo model file was created."
)

print(
    "\nMATERNAL HEALTH RISK ANALYSIS COMPLETE."
)

# ============================================================
# 16. HYPERPARAMETER TUNING
# ============================================================

print("\n\n" + "=" * 80)
print("16. CLEAN DATASET HYPERPARAMETER TUNING")
print("=" * 80)

print(
    "\nIMPORTANT:"
    "\nThe CLEAN test set remains untouched."
    "\nHyperparameter tuning uses TRAIN + VALIDATION only."
    "\nThe optimization metric is group-aware CV Macro-F1."
)


from sklearn.model_selection import GridSearchCV


# ============================================================
# USE CLEAN TRAIN + VALIDATION ONLY
# ============================================================

X_tune = clean_split[
    "X_train_val"
]

y_tune = clean_split[
    "y_train_val"
]

groups_tune = clean_split[
    "groups_train_val"
]

X_final_test = clean_split[
    "X_test"
]

y_final_test = clean_split[
    "y_test"
]


# ============================================================
# GROUP-AWARE CV
# ============================================================

tuning_cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# RANDOM FOREST SEARCH
# ============================================================

print("\n" + "-" * 80)
print("TUNING RANDOM FOREST")
print("-" * 80)

rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf_params = {
    "n_estimators": [
        300,
        500,
        800,
    ],

    "max_depth": [
        None,
        8,
        12,
        16,
    ],

    "min_samples_split": [
        2,
        4,
        8,
    ],

    "min_samples_leaf": [
        1,
        2,
        4,
    ],

    "max_features": [
        "sqrt",
        0.8,
    ],

    "class_weight": [
        "balanced",
        "balanced_subsample",
    ],
}


rf_search = GridSearchCV(
    estimator=rf,
    param_grid=rf_params,
    scoring="f1_macro",
    cv=tuning_cv,
    n_jobs=-1,
    refit=True,
    verbose=1,
)


rf_search.fit(
    X_tune,
    y_tune,
    groups=groups_tune,
)


print("\nBest Random Forest parameters:")

print(
    rf_search.best_params_
)

print(
    f"\nBest Random Forest CV Macro F1: "
    f"{rf_search.best_score_:.4f}"
)


# ============================================================
# EXTRA TREES SEARCH
# ============================================================

print("\n" + "-" * 80)
print("TUNING EXTRA TREES")
print("-" * 80)

et = ExtraTreesClassifier(
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

et_params = {
    "n_estimators": [
        300,
        500,
        800,
    ],

    "max_depth": [
        None,
        8,
        12,
        16,
    ],

    "min_samples_split": [
        2,
        4,
        8,
    ],

    "min_samples_leaf": [
        1,
        2,
        4,
    ],

    "max_features": [
        "sqrt",
        0.8,
    ],

    "class_weight": [
        "balanced",
        "balanced_subsample",
    ],
}


et_search = GridSearchCV(
    estimator=et,
    param_grid=et_params,
    scoring="f1_macro",
    cv=tuning_cv,
    n_jobs=-1,
    refit=True,
    verbose=1,
)


et_search.fit(
    X_tune,
    y_tune,
    groups=groups_tune,
)


print("\nBest Extra Trees parameters:")

print(
    et_search.best_params_
)

print(
    f"\nBest Extra Trees CV Macro F1: "
    f"{et_search.best_score_:.4f}"
)


# ============================================================
# XGBOOST SEARCH
# ============================================================

print("\n" + "-" * 80)
print("TUNING XGBOOST")
print("-" * 80)

xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    tree_method="hist",
    device="cuda",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

xgb_params = {
    "n_estimators": [
        200,
        400,
        600,
    ],

    "max_depth": [
        2,
        3,
        4,
        5,
    ],

    "learning_rate": [
        0.02,
        0.05,
        0.10,
    ],

    "min_child_weight": [
        1,
        3,
        5,
    ],

    "subsample": [
        0.8,
        1.0,
    ],

    "colsample_bytree": [
        0.8,
        1.0,
    ],

    "reg_alpha": [
        0.0,
        0.1,
        0.5,
    ],

    "reg_lambda": [
        1.0,
        2.0,
        5.0,
    ],
}


# XGBoost has a very large possible grid.
# RandomizedSearchCV gives us controlled exploration.

from sklearn.model_selection import RandomizedSearchCV


xgb_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=xgb_params,
    n_iter=30,
    scoring="f1_macro",
    cv=tuning_cv,
    n_jobs=1,
    refit=True,
    verbose=1,
    random_state=RANDOM_STATE,
)


xgb_search.fit(
    X_tune,
    y_tune,
    groups=groups_tune,
)


print("\nBest XGBoost parameters:")

print(
    xgb_search.best_params_
)

print(
    f"\nBest XGBoost CV Macro F1: "
    f"{xgb_search.best_score_:.4f}"
)


# ============================================================
# COMPARE TUNED MODELS
# ============================================================

print("\n" + "=" * 80)
print("17. TUNED MODEL COMPARISON")
print("=" * 80)


tuned_models = [
    (
        "RandomForest",
        rf_search.best_estimator_,
        rf_search.best_score_,
    ),

    (
        "ExtraTrees",
        et_search.best_estimator_,
        et_search.best_score_,
    ),

    (
        "XGBoost",
        xgb_search.best_estimator_,
        xgb_search.best_score_,
    ),
]


tuned_models.sort(
    key=lambda item: item[2],
    reverse=True,
)


print(
    "\n"
    f"{'Model':<18}"
    f"{'CV Macro F1':>15}"
)

print(
    "-" * 35
)

for (
    name,
    model,
    score,
) in tuned_models:

    print(
        f"{name:<18}"
        f"{score:>15.4f}"
    )


# ============================================================
# SELECT TUNED MODEL
# ============================================================

best_tuned_name = tuned_models[0][0]

best_tuned_model = tuned_models[0][1]

best_tuned_cv_f1 = tuned_models[0][2]


print(
    f"\nSelected tuned model: "
    f"{best_tuned_name}"
)

print(
    f"Best group-aware CV Macro F1: "
    f"{best_tuned_cv_f1:.4f}"
)


# ============================================================
# LOCKED CLEAN TEST EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("18. TUNED MODEL - LOCKED CLEAN TEST")
print("=" * 80)

print(
    "\nThe test set has not been used during tuning."
)

final_predictions = (
    best_tuned_model.predict(
        X_final_test
    )
)


final_accuracy = accuracy_score(
    y_final_test,
    final_predictions,
)

final_precision = precision_score(
    y_final_test,
    final_predictions,
    average="macro",
    zero_division=0,
)

final_recall = recall_score(
    y_final_test,
    final_predictions,
    average="macro",
    zero_division=0,
)

final_macro_f1 = f1_score(
    y_final_test,
    final_predictions,
    average="macro",
    zero_division=0,
)


print(
    f"\nFinal tuned model: "
    f"{best_tuned_name}"
)

print(
    f"\nTest Accuracy : "
    f"{final_accuracy:.4f}"
)

print(
    f"Test Precision: "
    f"{final_precision:.4f}"
)

print(
    f"Test Recall   : "
    f"{final_recall:.4f}"
)

print(
    f"Test Macro F1 : "
    f"{final_macro_f1:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_final_test,
        final_predictions,
    )
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_final_test,
        final_predictions,
        labels=[0, 1, 2],
        target_names=[
            "low risk",
            "mid risk",
            "high risk",
        ],
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# MID-RISK RECALL
# ============================================================

class_recalls = recall_score(
    y_final_test,
    final_predictions,
    average=None,
    labels=[0, 1, 2],
    zero_division=0,
)

print(
    "\nClass-specific recall:"
)

print(
    f"  Low risk : "
    f"{class_recalls[0]:.4f}"
)

print(
    f"  Mid risk : "
    f"{class_recalls[1]:.4f}"
)

print(
    f"  High risk: "
    f"{class_recalls[2]:.4f}"
)


# ============================================================
# FINAL TUNING SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("19. HYPERPARAMETER TUNING SUMMARY")
print("=" * 80)

print(
    f"""
Dataset:
    CLEAN maternal dataset

Training samples used for tuning:
    {len(X_tune)}

Locked test samples:
    {len(X_final_test)}

Selected model:
    {best_tuned_name}

Best group-aware CV Macro F1:
    {best_tuned_cv_f1:.4f}

Locked Test Accuracy:
    {final_accuracy:.4f}

Locked Test Macro F1:
    {final_macro_f1:.4f}

Low-risk recall:
    {class_recalls[0]:.4f}

Mid-risk recall:
    {class_recalls[1]:.4f}

High-risk recall:
    {class_recalls[2]:.4f}
"""
)


print(
    "No model file was saved."
)

print(
    "TUNING COMPLETE."
)

print("=" * 80)