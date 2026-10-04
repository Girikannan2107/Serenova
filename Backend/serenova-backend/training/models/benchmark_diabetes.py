from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"

RANDOM_STATE = 42
TEST_SIZE = 116
VAL_SIZE = 115

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


# =============================================================================
# HELPERS
# =============================================================================

def header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def make_preprocessor(scale=False):
    steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=False,
            ),
        )
    ]

    if scale:
        steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    return Pipeline(steps)


def build_models():
    models = {}

    models["Logistic Regression"] = Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(scale=True),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    models["Random Forest"] = Pipeline(
        [
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
    )

    models["Extra Trees"] = Pipeline(
        [
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
    )

    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline(
            [
                (
                    "preprocess",
                    make_preprocessor(scale=False),
                ),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    else:
        print("\nWARNING: XGBoost is not installed. Skipping XGBoost.")

    models["MLP"] = Pipeline(
        [
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
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=1500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return models


def evaluate(model, X, y):
    predictions = model.predict(X)

    return {
        "accuracy": accuracy_score(y, predictions),
        "macro_precision": precision_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    header("SERENOVA — DIABETES BASELINE BENCHMARK")

    print(f"Dataset: {RAW_PATH}")

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{RAW_PATH}"
        )

    df = pd.read_csv(RAW_PATH)

    # -------------------------------------------------------------------------
    # BASIC VALIDATION
    # -------------------------------------------------------------------------

    required = FEATURE_COLUMNS + [TARGET]

    missing_columns = [
        column for column in required
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Raw dataset unexpectedly contains missing values."
        )

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            "Outcome must contain only 0 and 1."
        )

    # -------------------------------------------------------------------------
    # ZERO -> MISSING POLICY
    # -------------------------------------------------------------------------

    header("1. PREPROCESSING POLICY")

    print("Zero values converted to missing values in:")

    for column in ZERO_AS_MISSING_COLUMNS:
        print(f"  - {column}")

    print("\nPregnancies=0 is retained as a valid value.")

    print(
        "\nMissing values are median-imputed inside each training pipeline."
    )

    print(
        "Imputation is fitted only on the training data."
    )

    # -------------------------------------------------------------------------
    # PREPARE X / y
    # -------------------------------------------------------------------------

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(np.int64).copy()

    for column in ZERO_AS_MISSING_COLUMNS:
        X.loc[X[column] == 0, column] = np.nan

    print("\nMissing values created by zero conversion:")

    for column in ZERO_AS_MISSING_COLUMNS:
        count = int(X[column].isna().sum())
        print(f"  {column:30s}: {count}")

    # -------------------------------------------------------------------------
    # LOCKED SPLIT
    #
    # 768 -> train 537, validation 115, test 116
    # -------------------------------------------------------------------------

    header("2. LEAKAGE-SAFE DATA SPLIT")

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

    print("\nClass distributions:")

    for name, labels in [
        ("Train", y_train),
        ("Validation", y_val),
        ("Test", y_test),
    ]:
        counts = labels.value_counts().sort_index()
        percentages = labels.value_counts(
            normalize=True
        ).sort_index() * 100

        print(f"\n{name}")

        for cls in [0, 1]:
            print(
                f"  Class {cls}: "
                f"{counts.get(cls, 0)} "
                f"({percentages.get(cls, 0):.2f}%)"
            )

    # -------------------------------------------------------------------------
    # INTEGRITY CHECK
    # -------------------------------------------------------------------------

    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    test_indices = set(X_test.index)

    assert train_indices.isdisjoint(val_indices)
    assert train_indices.isdisjoint(test_indices)
    assert val_indices.isdisjoint(test_indices)

    print("\nTrain/validation/test row leakage: PASS")

    # -------------------------------------------------------------------------
    # MODELS
    # -------------------------------------------------------------------------

    models = build_models()

    # -------------------------------------------------------------------------
    # VALIDATION BENCHMARK
    # -------------------------------------------------------------------------

    header("3. VALIDATION BENCHMARK")

    validation_results = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        metrics = evaluate(
            model,
            X_val,
            y_val,
        )

        validation_results[name] = metrics

        print(
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"Macro Precision={metrics['macro_precision']:.4f} | "
            f"Macro Recall={metrics['macro_recall']:.4f} | "
            f"Macro F1={metrics['macro_f1']:.4f}"
        )

    # -------------------------------------------------------------------------
    # CROSS-VALIDATION
    # -------------------------------------------------------------------------

    header("4. 5-FOLD CROSS-VALIDATION")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_results = {}

    for name, model in models.items():

        print(f"\nCV: {name}")

        scores = cross_val_score(
            model,
            X_train_full,
            y_train_full,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
        )

        cv_results[name] = scores

        print(
            f"Fold macro F1: "
            f"{np.round(scores, 4)}"
        )

        print(
            f"Mean macro F1: "
            f"{scores.mean():.4f}"
        )

        print(
            f"Std macro F1: "
            f"{scores.std():.4f}"
        )

    # -------------------------------------------------------------------------
    # MODEL SELECTION
    # -------------------------------------------------------------------------

    header("5. MODEL SELECTION")

    ranking = sorted(
        cv_results.items(),
        key=lambda item: item[1].mean(),
        reverse=True,
    )

    for rank, (name, scores) in enumerate(
        ranking,
        start=1,
    ):
        print(
            f"{rank}. {name:22s} "
            f"CV Macro F1={scores.mean():.4f} "
            f"+/- {scores.std():.4f}"
        )

    selected_name = ranking[0][0]

    print(
        f"\nSelected by CV macro F1: {selected_name}"
    )

    # -------------------------------------------------------------------------
    # LOCKED TEST
    # -------------------------------------------------------------------------

    header("6. LOCKED TEST EVALUATION")

    selected_model = models[selected_name]

    # Train selected model on train + validation.
    selected_model.fit(
        X_train_full,
        y_train_full,
    )

    test_predictions = selected_model.predict(X_test)

    test_metrics = {
        "accuracy": accuracy_score(
            y_test,
            test_predictions,
        ),
        "macro_precision": precision_score(
            y_test,
            test_predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_test,
            test_predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_test,
            test_predictions,
            average="macro",
            zero_division=0,
        ),
    }

    print(
        f"Accuracy:         {test_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro Precision:  {test_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall:     {test_metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1:         {test_metrics['macro_f1']:.4f}"
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
    # TEST INTEGRITY
    # -------------------------------------------------------------------------

    header("7. EVALUATION INTEGRITY")

    print("Test set used during preprocessing fitting: NO")
    print("Test set used during CV: NO")
    print("Test set used during model selection: NO")
    print("Test set used only for final evaluation: YES")

    print("\nNo model artifact saved.")
    print("Diabetes baseline benchmark complete.")


if __name__ == "__main__":
    main()