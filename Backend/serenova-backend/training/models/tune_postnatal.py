"""
Serenova - Postnatal Model Tuning

Purpose
-------
Tune Extra Trees, Random Forest, and XGBoost for the cleaned
postnatal binary classification problem.

Important methodology
---------------------
1. Missing rows are removed.
2. Exact duplicate rows are removed.
3. "Not interested to say" is treated as unknown and removed.
4. Timestamp is excluded from prediction.
5. Conflicting feature groups are removed.
6. Repeated feature combinations are kept as groups.
7. Train/validation/test splitting is group-aware.
8. Hyperparameter selection uses ONLY train + validation.
9. The locked test set is used only once for final evaluation.
10. Preprocessing is fitted inside the model pipeline.
11. No model is saved.

Current baseline
----------------
Extra Trees:
    CV Macro F1  = 0.7356
    Test Macro F1 = 0.6927
    Test Yes Recall = 0.4800
    Test Yes F1 = 0.5854
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


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

RANDOM_STATE = 42

N_SPLITS = 5

# These match the completed postnatal benchmark.
TEST_FOLD = 4
VAL_FOLD = 3

# Current benchmark baseline.
BASELINE_CV_F1 = 0.7356
BASELINE_TEST_F1 = 0.6927
BASELINE_TEST_YES_RECALL = 0.4800
BASELINE_TEST_YES_F1 = 0.5854


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
# ONE-HOT ENCODER
# ============================================================

def make_encoder():
    """
    Compatibility with different scikit-learn versions.
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


# ============================================================
# MODEL PIPELINE
# ============================================================

def make_pipeline(model):
    """
    Encoder is inside the pipeline.

    Therefore, during CV:
        encoder.fit()
    is performed only on that fold's training data.

    This prevents preprocessing leakage.
    """

    return Pipeline(
        steps=[
            (
                "encoder",
                make_encoder(),
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# LOAD + CLEAN DATA
# ============================================================

def load_and_clean():

    print("=" * 80)
    print("POSTNATAL DATA LOADING AND CLEANING")
    print("=" * 80)

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_PATH}"
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
    # Remove rows containing missing values
    # --------------------------------------------------------

    missing_rows = int(
        df.isna()
        .any(axis=1)
        .sum()
    )

    print(
        f"\nRows containing missing values: "
        f"{missing_rows}"
    )

    df = (
        df
        .dropna()
        .copy()
    )

    print(
        f"After missing-row removal: "
        f"{df.shape}"
    )

    # --------------------------------------------------------
    # Remove exact duplicates
    # --------------------------------------------------------

    duplicate_count = int(
        df.duplicated()
        .sum()
    )

    print(
        f"\nExact duplicate rows removed: "
        f"{duplicate_count}"
    )

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        f"After duplicate removal: "
        f"{df.shape}"
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print(
        "\nTarget distribution before unknown removal:"
    )

    print(
        df[TARGET_COLUMN]
        .value_counts(
            dropna=False
        )
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

    unknown_count = int(
        unknown_mask.sum()
    )

    print(
        "\nUnknown target rows removed "
        '("Not interested to say"): '
        f"{unknown_count}"
    )

    df = df.loc[
        ~unknown_mask
    ].copy()

    # --------------------------------------------------------
    # Convert target to binary
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
            "Unexpected target values remain "
            "after binary target mapping."
        )

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(np.int64)
    )

    # --------------------------------------------------------
    # Detect conflicting feature groups
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
        "\nConflicting feature groups removed: "
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
            "Rows removed because of conflicts: "
            f"{conflict_rows}"
        )

        df = df.loc[
            ~conflict_mask
        ].copy()

    # --------------------------------------------------------
    # Rebuild final X/y
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
    # Build group IDs
    #
    # Same feature combination = same group.
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
    # Integrity checks
    # --------------------------------------------------------

    if X_raw.isna().any().any():

        raise AssertionError(
            "Missing feature values remain."
        )

    if y.isna().any():

        raise AssertionError(
            "Missing target values remain."
        )

    if not set(
        y.unique()
    ).issubset(
        {0, 1}
    ):

        raise AssertionError(
            "Target contains values other than 0 and 1."
        )

    if len(X_raw) != len(y):

        raise AssertionError(
            "X and y lengths do not match."
        )

    print(
        "\nFinal cleaned dataset:"
    )

    print(
        f"  Rows: {len(df)}"
    )

    print(
        f"  Features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        f"  Unique feature groups: "
        f"{len(np.unique(group_ids))}"
    )

    print(
        "\nTarget distribution:"
    )

    counts = (
        y.value_counts()
        .sort_index()
    )

    for label, count in counts.items():

        label_name = (
            "No"
            if int(label) == 0
            else "Yes"
        )

        percentage = (
            100.0
            * count
            / len(y)
        )

        print(
            f"  {label_name}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    return (
        X_raw,
        y,
        group_ids,
    )


# ============================================================
# LOCKED SPLIT
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
    # Locked test fold
    # --------------------------------------------------------

    train_val_idx, test_idx = (
        folds[TEST_FOLD]
    )

    # --------------------------------------------------------
    # Train/validation split
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
# PARAMETER SPACES
# ============================================================

def build_parameter_spaces():

    return {

        # ----------------------------------------------------
        # Extra Trees
        # ----------------------------------------------------

        "Extra Trees": {

            "n_estimators": [
                300,
                500,
            ],

            "max_depth": [
                None,
                6,
                10,
                14,
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
                1.0,
            ],

            "class_weight": [
                "balanced",
                "balanced_subsample",
            ],
        },

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        "Random Forest": {

            "n_estimators": [
                300,
                500,
            ],

            "max_depth": [
                None,
                6,
                10,
                14,
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
                1.0,
            ],

            "class_weight": [
                "balanced",
                "balanced_subsample",
            ],
        },

        # ----------------------------------------------------
        # XGBoost
        #
        # Deliberately smaller than the original search plan.
        # The dataset is only 314 rows.
        # ----------------------------------------------------

        "XGBoost": {

            "n_estimators": [
                100,
                200,
                300,
            ],

            "max_depth": [
                2,
                3,
                4,
            ],

            "learning_rate": [
                0.02,
                0.05,
                0.1,
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

            "reg_lambda": [
                1.0,
                2.0,
                5.0,
            ],
        },
    }


# ============================================================
# MODEL FACTORY
# ============================================================

def make_model(
    model_name,
    params,
):

    if model_name == "Extra Trees":

        estimator = (
            ExtraTreesClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **params,
            )
        )

    elif model_name == "Random Forest":

        estimator = (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **params,
            )
        )

    elif model_name == "XGBoost":

        estimator = (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **params,
            )
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return make_pipeline(
        estimator
    )


# ============================================================
# CROSS-VALIDATION
# ============================================================

def evaluate_cv(
    model,
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

    macro_f1_scores = []
    yes_recall_scores = []
    yes_f1_scores = []
    accuracy_scores = []

    for (
        train_idx,
        validation_idx,
    ) in splitter.split(
        X,
        y,
        groups,
    ):

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
        )

        predictions = (
            model.predict(
                X.iloc[
                    validation_idx
                ]
            )
        )

        macro_f1_scores.append(
            f1_score(
                y.iloc[
                    validation_idx
                ],
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        yes_recall_scores.append(
            recall_score(
                y.iloc[
                    validation_idx
                ],
                predictions,
                pos_label=1,
                zero_division=0,
            )
        )

        yes_f1_scores.append(
            f1_score(
                y.iloc[
                    validation_idx
                ],
                predictions,
                pos_label=1,
                zero_division=0,
            )
        )

        accuracy_scores.append(
            accuracy_score(
                y.iloc[
                    validation_idx
                ],
                predictions,
            )
        )

    return {
        "macro_f1_mean": float(
            np.mean(
                macro_f1_scores
            )
        ),

        "macro_f1_std": float(
            np.std(
                macro_f1_scores
            )
        ),

        "yes_recall_mean": float(
            np.mean(
                yes_recall_scores
            )
        ),

        "yes_f1_mean": float(
            np.mean(
                yes_f1_scores
            )
        ),

        "accuracy_mean": float(
            np.mean(
                accuracy_scores
            )
        ),
    }


# ============================================================
# TUNE ONE MODEL
# ============================================================

def tune_model(
    model_name,
    parameter_space,
    X,
    y,
    groups,
):

    parameter_grid = list(
        ParameterGrid(
            parameter_space
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"TUNING {model_name.upper()}"
    )

    print(
        "=" * 80
    )

    print(
        f"Candidates: "
        f"{len(parameter_grid)}"
    )

    print(
        f"CV fits: "
        f"{len(parameter_grid) * N_SPLITS}"
    )

    results = []

    best_score = -np.inf

    for (
        candidate_number,
        params,
    ) in enumerate(
        parameter_grid,
        start=1,
    ):

        model = make_model(
            model_name,
            params,
        )

        scores = evaluate_cv(
            model=model,
            X=X,
            y=y,
            groups=groups,
        )

        result = {
            "model": model_name,
            "candidate": candidate_number,
            "macro_f1_mean": scores[
                "macro_f1_mean"
            ],
            "macro_f1_std": scores[
                "macro_f1_std"
            ],
            "yes_recall_mean": scores[
                "yes_recall_mean"
            ],
            "yes_f1_mean": scores[
                "yes_f1_mean"
            ],
            "accuracy_mean": scores[
                "accuracy_mean"
            ],
            "params": params,
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # Primary selection:
        # Macro F1
        #
        # Tie breakers:
        # Yes recall
        # Yes F1
        # ----------------------------------------------------

        current_key = (
            scores["macro_f1_mean"],
            scores["yes_recall_mean"],
            scores["yes_f1_mean"],
        )

        if candidate_number == 1:

            best_key = current_key
            is_new_best = True

        else:

            is_new_best = (
                current_key
                > best_key
            )

        if is_new_best:

            best_key = current_key

            best_score = (
                scores["macro_f1_mean"]
            )

            print(
                "\nNew best candidate "
                f"{candidate_number}/"
                f"{len(parameter_grid)}"
            )

            print(
                f"  Macro F1: "
                f"{scores['macro_f1_mean']:.4f}"
            )

            print(
                f"  Macro F1 Std: "
                f"{scores['macro_f1_std']:.4f}"
            )

            print(
                f"  Yes recall: "
                f"{scores['yes_recall_mean']:.4f}"
            )

            print(
                f"  Yes F1: "
                f"{scores['yes_f1_mean']:.4f}"
            )

            print(
                f"  Accuracy: "
                f"{scores['accuracy_mean']:.4f}"
            )

            print(
                f"  Params: {params}"
            )

    results_df = (
        pd.DataFrame(
            results
        )
        .sort_values(
            [
                "macro_f1_mean",
                "yes_recall_mean",
                "yes_f1_mean",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return results_df


# ============================================================
# LOCKED TEST EVALUATION
# ============================================================

def evaluate_test(
    model,
    model_name,
    X_train_val,
    y_train_val,
    X_test,
    y_test,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"LOCKED TEST EVALUATION - "
        f"{model_name}"
    )

    print(
        "=" * 80
    )

    print(
        "\nFitting selected model on "
        "train + validation only."
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

    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "yes_precision": yes_precision,
        "yes_recall": yes_recall,
        "yes_f1": yes_f1,
        "confusion_matrix": cm,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SERENOVA POSTNATAL MODEL TUNING")
    print("=" * 80)

    # ========================================================
    # 1. LOAD + CLEAN
    # ========================================================

    (
        X,
        y,
        groups,
    ) = load_and_clean()

    # ========================================================
    # 2. CREATE LOCKED SPLIT
    # ========================================================

    (
        train_idx,
        val_idx,
        test_idx,
    ) = create_locked_split(
        X,
        y,
        groups,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TEST
    #
    # This was the missing variable in the previous version.
    # --------------------------------------------------------

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

    # ========================================================
    # 3. TRAIN + VALIDATION DATA
    # ========================================================

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

    groups_train_val = np.concatenate(
        [
            groups_train,
            groups_val,
        ]
    )

    # ========================================================
    # 4. SPLIT INTEGRITY
    # ========================================================

    train_val_overlap = (
        set(groups_train)
        .intersection(
            set(groups_val)
        )
    )

    train_test_overlap = (
        set(groups_train)
        .intersection(
            set(groups_test)
        )
    )

    val_test_overlap = (
        set(groups_val)
        .intersection(
            set(groups_test)
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LOCKED DATA SPLIT"
    )

    print(
        "=" * 80
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

    print(
        "\nGroup leakage:"
    )

    print(
        "  Train-Val:  "
        + (
            "FAIL"
            if train_val_overlap
            else "PASS"
        )
    )

    print(
        "  Train-Test: "
        + (
            "FAIL"
            if train_test_overlap
            else "PASS"
        )
    )

    print(
        "  Val-Test:   "
        + (
            "FAIL"
            if val_test_overlap
            else "PASS"
        )
    )

    if train_val_overlap:
        raise AssertionError(
            "Train/validation group leakage detected."
        )

    if train_test_overlap:
        raise AssertionError(
            "Train/test group leakage detected."
        )

    if val_test_overlap:
        raise AssertionError(
            "Validation/test group leakage detected."
        )

    # ========================================================
    # 5. PARAMETER SPACES
    # ========================================================

    parameter_spaces = (
        build_parameter_spaces()
    )

    # ========================================================
    # 6. TUNE MODELS
    #
    # ONLY X_train_val is passed.
    #
    # X_test is not passed to tuning.
    # ========================================================

    tuning_results = {}

    for model_name in [
        "Extra Trees",
        "Random Forest",
        "XGBoost",
    ]:

        tuning_results[
            model_name
        ] = tune_model(
            model_name=model_name,
            parameter_space=(
                parameter_spaces[
                    model_name
                ]
            ),
            X=X_train_val,
            y=y_train_val,
            groups=groups_train_val,
        )

    # ========================================================
    # 7. BEST CANDIDATE FOR EACH MODEL
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BEST TUNED CANDIDATES"
    )

    print(
        "=" * 80
    )

    best_candidates = {}

    for (
        model_name,
        results_df,
    ) in tuning_results.items():

        best = results_df.iloc[0]

        best_candidates[
            model_name
        ] = best

        print(
            f"\n{model_name}"
        )

        print(
            f"  CV Macro F1: "
            f"{best['macro_f1_mean']:.4f}"
        )

        print(
            f"  CV Macro F1 Std: "
            f"{best['macro_f1_std']:.4f}"
        )

        print(
            f"  CV Yes recall: "
            f"{best['yes_recall_mean']:.4f}"
        )

        print(
            f"  CV Yes F1: "
            f"{best['yes_f1_mean']:.4f}"
        )

        print(
            f"  CV Accuracy: "
            f"{best['accuracy_mean']:.4f}"
        )

        print(
            "  Parameters:"
        )

        for (
            key,
            value,
        ) in best[
            "params"
        ].items():

            print(
                f"    {key}: {value}"
            )

    # ========================================================
    # 8. SELECT OVERALL TUNED MODEL
    # ========================================================

    ordered_candidates = sorted(
        best_candidates.items(),
        key=lambda item: (
            item[1][
                "macro_f1_mean"
            ],
            item[1][
                "yes_recall_mean"
            ],
            item[1][
                "yes_f1_mean"
            ],
        ),
        reverse=True,
    )

    selected_name = (
        ordered_candidates[0][0]
    )

    selected_candidate = (
        ordered_candidates[0][1]
    )

    selected_params = (
        selected_candidate[
            "params"
        ]
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OVERALL TUNED MODEL"
    )

    print(
        "=" * 80
    )

    print(
        f"\nSelected model: "
        f"{selected_name}"
    )

    print(
        f"CV Macro F1: "
        f"{selected_candidate['macro_f1_mean']:.4f}"
    )

    print(
        f"CV Yes recall: "
        f"{selected_candidate['yes_recall_mean']:.4f}"
    )

    print(
        f"CV Yes F1: "
        f"{selected_candidate['yes_f1_mean']:.4f}"
    )

    print(
        "\nSelected parameters:"
    )

    for (
        key,
        value,
    ) in selected_params.items():

        print(
            f"  {key}: {value}"
        )

    # ========================================================
    # 9. LOCKED TEST
    #
    # FIRST time the test set is used.
    # ========================================================

    selected_model = make_model(
        selected_name,
        selected_params,
    )

    test_result = evaluate_test(
        model=selected_model,
        model_name=selected_name,
        X_train_val=X_train_val,
        y_train_val=y_train_val,
        X_test=X_test,
        y_test=y_test,
    )

    # ========================================================
    # 10. BASELINE VS TUNED
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BASELINE VS TUNED COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        "\nCURRENT BASELINE"
    )

    print(
        f"  Extra Trees CV Macro F1: "
        f"{BASELINE_CV_F1:.4f}"
    )

    print(
        f"  Extra Trees Test Macro F1: "
        f"{BASELINE_TEST_F1:.4f}"
    )

    print(
        f"  Extra Trees Test Yes recall: "
        f"{BASELINE_TEST_YES_RECALL:.4f}"
    )

    print(
        f"  Extra Trees Test Yes F1: "
        f"{BASELINE_TEST_YES_F1:.4f}"
    )

    print(
        "\nTUNED RESULT"
    )

    print(
        f"  Model: "
        f"{selected_name}"
    )

    print(
        f"  CV Macro F1: "
        f"{selected_candidate['macro_f1_mean']:.4f}"
    )

    print(
        f"  Test Macro F1: "
        f"{test_result['macro_f1']:.4f}"
    )

    print(
        f"  Test Yes recall: "
        f"{test_result['yes_recall']:.4f}"
    )

    print(
        f"  Test Yes F1: "
        f"{test_result['yes_f1']:.4f}"
    )

    cv_delta = (
        selected_candidate[
            "macro_f1_mean"
        ]
        - BASELINE_CV_F1
    )

    test_delta = (
        test_result[
            "macro_f1"
        ]
        - BASELINE_TEST_F1
    )

    yes_recall_delta = (
        test_result[
            "yes_recall"
        ]
        - BASELINE_TEST_YES_RECALL
    )

    yes_f1_delta = (
        test_result[
            "yes_f1"
        ]
        - BASELINE_TEST_YES_F1
    )

    print(
        "\nDELTA"
    )

    print(
        f"  CV Macro F1: "
        f"{cv_delta:+.4f}"
    )

    print(
        f"  Test Macro F1: "
        f"{test_delta:+.4f}"
    )

    print(
        f"  Test Yes recall: "
        f"{yes_recall_delta:+.4f}"
    )

    print(
        f"  Test Yes F1: "
        f"{yes_f1_delta:+.4f}"
    )

    # ========================================================
    # 11. TUNING DECISION
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TUNING DECISION"
    )

    print(
        "=" * 80
    )

    improved_cv = (
        selected_candidate[
            "macro_f1_mean"
        ]
        > BASELINE_CV_F1
    )

    improved_test = (
        test_result[
            "macro_f1"
        ]
        > BASELINE_TEST_F1
    )

    improved_yes_recall = (
        test_result[
            "yes_recall"
        ]
        > BASELINE_TEST_YES_RECALL
    )

    if (
        improved_cv
        and improved_test
        and improved_yes_recall
    ):

        print(
            "\nTuning improved:"
        )

        print(
            "  - CV macro F1"
        )

        print(
            "  - locked-test macro F1"
        )

        print(
            "  - Yes recall"
        )

        print(
            "\nThe tuned configuration is "
            "the stronger experimental candidate."
        )

    elif (
        improved_cv
        and improved_test
    ):

        print(
            "\nTuning improved CV macro F1 "
            "and locked-test macro F1."
        )

        print(
            "However, Yes recall did not improve."
        )

        print(
            "\nTreat this as an experimental "
            "improvement, not a final clinical model."
        )

    elif improved_cv:

        print(
            "\nTuning improved CV macro F1 "
            "but did not improve locked-test "
            "macro F1."
        )

        print(
            "This suggests limited evidence "
            "of improved generalization."
        )

        print(
            "\nKeep the simpler baseline "
            "as the benchmark."
        )

    else:

        print(
            "\nTuning did not improve "
            "CV macro F1."
        )

        print(
            "The current Extra Trees baseline "
            "remains the benchmark."
        )

    print(
        "\nNo model was saved."
    )

    # ========================================================
    # 12. FINAL INTEGRITY
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "EVALUATION INTEGRITY"
    )

    print(
        "=" * 80
    )

    checks = {
        "No missing values":
            not X.isna().any().any(),

        "Binary target":
            set(y.unique()).issubset(
                {0, 1}
            ),

        "No train-val leakage":
            not bool(
                train_val_overlap
            ),

        "No train-test leakage":
            not bool(
                train_test_overlap
            ),

        "No val-test leakage":
            not bool(
                val_test_overlap
            ),

        "Test excluded from tuning":
            True,

        "Preprocessing inside pipeline":
            True,

        "No model saved":
            True,
    }

    for (
        name,
        passed,
    ) in checks.items():

        print(
            f"  {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if not all(
        checks.values()
    ):

        raise AssertionError(
            "Evaluation integrity check failed."
        )

    print(
        "\nPostnatal tuning experiment complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()