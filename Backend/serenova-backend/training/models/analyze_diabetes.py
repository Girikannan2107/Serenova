from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"


EXPECTED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

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

ZERO_SUSPECT_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def check_finite(df: pd.DataFrame, columns):
    values = df[columns].to_numpy(dtype=np.float64)

    return bool(np.isfinite(values).all())


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header("SERENOVA — DIABETES DATA QUALITY ANALYSIS")

    print(f"Raw dataset: {RAW_PATH}")

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Diabetes dataset not found:\n{RAW_PATH}"
        )

    df = pd.read_csv(RAW_PATH)

    # -------------------------------------------------------------------------
    # BASIC SHAPE
    # -------------------------------------------------------------------------

    print_header("1. BASIC DATASET INFORMATION")

    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\nColumns:")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i:2d}. {column}")

    print("\nData types:")
    print(df.dtypes.to_string())

    # -------------------------------------------------------------------------
    # COLUMN VALIDATION
    # -------------------------------------------------------------------------

    print_header("2. COLUMN VALIDATION")

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    unexpected_columns = [
        column for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        print("Missing expected columns:")
        for column in missing_columns:
            print(f"  - {column}")
    else:
        print("Expected columns: PASS")

    if unexpected_columns:
        print("Unexpected columns:")
        for column in unexpected_columns:
            print(f"  - {column}")
    else:
        print("Unexpected columns: NONE")

    if missing_columns or unexpected_columns:
        raise ValueError("Column validation failed.")

    # -------------------------------------------------------------------------
    # MISSING VALUES
    # -------------------------------------------------------------------------

    print_header("3. MISSING VALUE ANALYSIS")

    missing_by_column = df.isna().sum()

    print(missing_by_column.to_string())

    total_missing = int(missing_by_column.sum())

    print(f"\nTotal missing cells: {total_missing}")
    print(
        "Missing values: PASS"
        if total_missing == 0
        else "Missing values: REVIEW REQUIRED"
    )

    # -------------------------------------------------------------------------
    # NUMERIC VALIDATION
    # -------------------------------------------------------------------------

    print_header("4. NUMERIC / FINITE VALUE VALIDATION")

    numeric_columns = EXPECTED_COLUMNS

    non_numeric = []

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            non_numeric.append(column)

    if non_numeric:
        print("Non-numeric columns detected:")
        for column in non_numeric:
            print(f"  - {column}")
        raise ValueError("Numeric validation failed.")

    finite_ok = check_finite(df, numeric_columns)

    print(
        "All numeric values finite: PASS"
        if finite_ok
        else "All numeric values finite: FAIL"
    )

    # -------------------------------------------------------------------------
    # TARGET
    # -------------------------------------------------------------------------

    print_header("5. TARGET ANALYSIS")

    target_counts = df["Outcome"].value_counts().sort_index()

    print("Outcome distribution:")
    print(target_counts.to_string())

    print("\nOutcome percentages:")
    target_percent = (
        df["Outcome"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
    )

    for label, percentage in target_percent.items():
        print(f"  Outcome {label}: {percentage:.2f}%")

    unique_targets = sorted(df["Outcome"].dropna().unique().tolist())

    print(f"\nUnique target values: {unique_targets}")

    if unique_targets == [0, 1]:
        print("Binary target validation: PASS")
    else:
        print("Binary target validation: REVIEW REQUIRED")

    # -------------------------------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # -------------------------------------------------------------------------

    print_header("6. FEATURE DESCRIPTIVE STATISTICS")

    print(
        df[FEATURE_COLUMNS]
        .describe()
        .transpose()
        .to_string()
    )

    # -------------------------------------------------------------------------
    # ZERO-VALUE ANALYSIS
    # -------------------------------------------------------------------------

    print_header("7. ZERO-VALUE ANALYSIS")

    print(
        "Zero values are NOT removed at this stage."
    )

    print(
        "They will be investigated before the modeling preprocessing "
        "policy is finalized."
    )

    for column in ZERO_SUSPECT_COLUMNS:

        zero_count = int((df[column] == 0).sum())
        zero_percentage = zero_count / len(df) * 100

        print(
            f"{column:30s} "
            f"zeros={zero_count:4d} "
            f"({zero_percentage:6.2f}%)"
        )

    # -------------------------------------------------------------------------
    # ZERO VALUES BY TARGET
    # -------------------------------------------------------------------------

    print_header("8. ZERO VALUES BY TARGET")

    for column in ZERO_SUSPECT_COLUMNS:

        print(f"\n{column}")

        zero_by_target = (
            df.loc[df[column] == 0, "Outcome"]
            .value_counts()
            .sort_index()
        )

        if zero_by_target.empty:
            print("  No zero values.")
        else:
            print(zero_by_target.to_string())

    # -------------------------------------------------------------------------
    # EXACT DUPLICATES
    # -------------------------------------------------------------------------

    print_header("9. EXACT DUPLICATE ANALYSIS")

    duplicate_mask = df.duplicated(keep=False)
    duplicate_participating = int(duplicate_mask.sum())

    duplicate_after_first = int(df.duplicated(keep="first").sum())

    print(
        f"Rows participating in exact duplicates: "
        f"{duplicate_participating}"
    )

    print(
        f"Exact duplicate rows after first occurrence: "
        f"{duplicate_after_first}"
    )

    # -------------------------------------------------------------------------
    # FEATURE COMBINATION ANALYSIS
    # -------------------------------------------------------------------------

    print_header("10. FEATURE-COMBINATION ANALYSIS")

    feature_index = pd.MultiIndex.from_frame(
        df[FEATURE_COLUMNS]
    )

    group_codes, unique_groups = pd.factorize(
        feature_index,
        sort=False
    )

    group_sizes = pd.Series(group_codes).value_counts()

    unique_feature_groups = int(len(unique_groups))

    repeated_groups = int((group_sizes > 1).sum())

    largest_group = int(group_sizes.max())

    print(f"Unique feature combinations: {unique_feature_groups}")
    print(f"Repeated feature combinations: {repeated_groups}")
    print(f"Largest repeated group size: {largest_group}")

    # -------------------------------------------------------------------------
    # CONFLICTING FEATURE GROUPS
    # -------------------------------------------------------------------------

    print_header("11. CONFLICTING FEATURE GROUP ANALYSIS")

    temp = df[FEATURE_COLUMNS + ["Outcome"]].copy()

    temp["_group_id"] = group_codes

    target_counts_by_group = (
        temp.groupby("_group_id")["Outcome"]
        .nunique()
    )

    conflicting_group_ids = target_counts_by_group[
        target_counts_by_group > 1
    ].index.tolist()

    print(
        f"Conflicting feature groups: "
        f"{len(conflicting_group_ids)}"
    )

    if conflicting_group_ids:

        conflict_rows = temp[
            temp["_group_id"].isin(conflicting_group_ids)
        ]

        print(
            f"Rows belonging to conflicting groups: "
            f"{len(conflict_rows)}"
        )

        print("\nConflict examples:")

        for group_id in conflicting_group_ids[:10]:

            group_rows = temp[
                temp["_group_id"] == group_id
            ]

            print(f"\nGroup {group_id}:")
            print(
                group_rows[
                    FEATURE_COLUMNS + ["Outcome"]
                ].to_string(index=False)
            )

    else:
        print("No conflicting feature groups found.")

    # -------------------------------------------------------------------------
    # GROUP SIZE DISTRIBUTION
    # -------------------------------------------------------------------------

    print_header("12. FEATURE-GROUP SIZE DISTRIBUTION")

    size_distribution = group_sizes.value_counts().sort_index()

    print("Group size -> number of groups")

    for size, count in size_distribution.items():
        print(
            f"  {int(size):3d} -> {int(count):4d}"
        )

    # -------------------------------------------------------------------------
    # FEATURE CORRELATION
    # -------------------------------------------------------------------------

    print_header("13. FEATURE / TARGET CORRELATION")

    correlation = df[
        FEATURE_COLUMNS + ["Outcome"]
    ].corr(numeric_only=True)["Outcome"].drop("Outcome")

    print(
        correlation
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .to_string()
    )

    # -------------------------------------------------------------------------
    # RANGE CHECKS
    # -------------------------------------------------------------------------

    print_header("14. RANGE / PLAUSIBILITY REVIEW")

    for column in FEATURE_COLUMNS:

        minimum = df[column].min()
        maximum = df[column].max()

        print(
            f"{column:30s} "
            f"min={minimum:10.4f} "
            f"max={maximum:10.4f}"
        )

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    print_header("15. SUMMARY")

    print(f"Rows: {len(df)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Target: Outcome")
    print(f"Target classes: {unique_targets}")
    print(f"Missing cells: {total_missing}")
    print(f"Exact duplicate rows after first: {duplicate_after_first}")
    print(f"Unique feature groups: {unique_feature_groups}")
    print(f"Repeated feature groups: {repeated_groups}")
    print(f"Conflicting feature groups: {len(conflicting_group_ids)}")

    print("\nImportant preprocessing decision:")
    print(
        "ZERO VALUES HAVE NOT BEEN CONVERTED TO MISSING VALUES YET."
    )

    print(
        "\nData-quality analysis complete."
    )

    print(
        "No rows were removed."
    )

    print(
        "No model was trained."
    )

    print(
        "No model artifact was saved."
    )


if __name__ == "__main__":
    main()