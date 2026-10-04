from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "post natal data.csv"
)

TARGET = "Suicide attempt"
TIMESTAMP_COLUMN = "Timestamp"

UNKNOWN_TARGET = "Not interested to say"

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("SERENOVA - POSTNATAL DATA CLEANING + LEAKAGE-SAFE SPLIT")
print("=" * 80)

print(f"\nCSV:")
print(CSV_PATH)


# ============================================================
# LOAD DATA
# ============================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

print("\n" + "=" * 80)
print("1. ORIGINAL DATASET")
print("=" * 80)

print(f"\nOriginal shape: {df.shape}")

print("\nColumns:")

for i, column in enumerate(df.columns):
    print(f"  {i}: {column}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    TIMESTAMP_COLUMN,
    "Age",
    "Feeling sad or Tearful",
    "Irritable towards baby & partner",
    "Trouble sleeping at night",
    "Problems concentrating or making decision",
    "Overeating or loss of appetite",
    "Feeling anxious",
    "Feeling of guilt",
    "Problems of bonding with baby",
    TARGET,
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_columns
        )
    )

print("\nPASS: All required columns found.")


# ============================================================
# ORIGINAL TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("2. ORIGINAL TARGET DISTRIBUTION")
print("=" * 80)

original_target_counts = (
    df[TARGET]
    .value_counts(dropna=False)
)

original_target_percent = (
    df[TARGET]
    .value_counts(
        normalize=True,
        dropna=False,
    )
    * 100
)

for label in original_target_counts.index:

    print(
        f"  {label}: "
        f"{original_target_counts[label]} "
        f"({original_target_percent[label]:.2f}%)"
    )


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("3. MISSING VALUE ANALYSIS")
print("=" * 80)

missing_counts = df.isna().sum()

print("\nMissing values by column:")

print(
    missing_counts[
        missing_counts > 0
    ]
)

missing_row_mask = df.isna().any(axis=1)

print(
    f"\nRows containing missing values: "
    f"{int(missing_row_mask.sum())}"
)


# ============================================================
# REMOVE MISSING ROWS
# ============================================================

original_row_count = len(df)

df = (
    df
    .dropna()
    .reset_index(drop=True)
)

removed_missing_rows = (
    original_row_count - len(df)
)

print(
    f"\nRows removed because of missing values: "
    f"{removed_missing_rows}"
)

print(
    f"Rows remaining: "
    f"{len(df)}"
)


# ============================================================
# EXACT DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("4. EXACT DUPLICATE ANALYSIS")
print("=" * 80)

duplicate_participating = int(
    df.duplicated(
        keep=False
    ).sum()
)

duplicate_rows_removed = int(
    df.duplicated(
        keep="first"
    ).sum()
)

print(
    f"\nRows participating in exact duplicates: "
    f"{duplicate_participating}"
)

print(
    f"Duplicate rows removed: "
    f"{duplicate_rows_removed}"
)


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

df = (
    df
    .drop_duplicates(
        keep="first"
    )
    .reset_index(drop=True)
)

print(
    f"\nRows remaining after duplicate removal: "
    f"{len(df)}"
)


# ============================================================
# TARGET CLEANING
# ============================================================

print("\n" + "=" * 80)
print("5. TARGET CLEANING")
print("=" * 80)

unknown_target_count = int(
    (
        df[TARGET]
        == UNKNOWN_TARGET
    ).sum()
)

print(
    f"\n'{UNKNOWN_TARGET}' rows: "
    f"{unknown_target_count}"
)

print(
    "\nThese rows will NOT be converted to 'No'."
)

print(
    "They will be excluded from binary model training."
)


# ============================================================
# REMOVE UNKNOWN TARGET
# ============================================================

df = df[
    df[TARGET]
    != UNKNOWN_TARGET
].reset_index(drop=True)

print(
    f"\nRows removed due to unknown target: "
    f"{unknown_target_count}"
)

print(
    f"Rows remaining: "
    f"{len(df)}"
)


# ============================================================
# BINARY TARGET ENCODING
# ============================================================

target_mapping = {
    "No": 0,
    "Yes": 1,
}

unexpected_targets = (
    set(
        df[TARGET]
        .dropna()
        .unique()
    )
    - set(
        target_mapping.keys()
    )
)

if unexpected_targets:

    raise ValueError(
        "Unexpected target values found:\n"
        f"{unexpected_targets}"
    )

df["target_binary"] = (
    df[TARGET]
    .map(target_mapping)
    .astype(np.int64)
)

print("\nBinary target mapping:")

print("  No  -> 0")
print("  Yes -> 1")


# ============================================================
# MODEL FEATURE COLUMNS
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column not in [
        TIMESTAMP_COLUMN,
        TARGET,
        "target_binary",
    ]
]

print("\n" + "=" * 80)
print("6. MODEL FEATURES")
print("=" * 80)

print(
    "\nTimestamp is excluded from model features."
)

print(
    f"\nFeature count: "
    f"{len(feature_columns)}"
)

for column in feature_columns:

    print(
        f"  - {column}"
    )


# ============================================================
# FEATURE GROUP ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("7. FEATURE GROUP ANALYSIS")
print("=" * 80)

group_counts = (
    df
    .groupby(
        feature_columns,
        dropna=False,
    )
    .size()
)

print(
    f"\nUnique feature combinations: "
    f"{len(group_counts)}"
)

print(
    f"Repeated feature combinations: "
    f"{int((group_counts > 1).sum())}"
)

print(
    f"Largest feature group: "
    f"{int(group_counts.max())}"
)


# ============================================================
# CONFLICTING LABEL ANALYSIS
# ============================================================

label_counts_per_group = (
    df
    .groupby(
        feature_columns,
        dropna=False,
    )["target_binary"]
    .nunique()
)

conflicting_groups = (
    label_counts_per_group[
        label_counts_per_group > 1
    ]
)

print(
    f"\nConflicting feature groups: "
    f"{len(conflicting_groups)}"
)


# ============================================================
# REMOVE CONFLICTING GROUPS
# ============================================================

conflicting_rows_removed = 0

if len(conflicting_groups) > 0:

    conflict_keys = set(
        conflicting_groups.index
    )

    row_keys = [
        tuple(row)
        for row in df[
            feature_columns
        ].to_numpy()
    ]

    conflict_mask = np.array(
        [
            key in conflict_keys
            for key in row_keys
        ],
        dtype=bool,
    )

    conflicting_rows_removed = int(
        conflict_mask.sum()
    )

    df = (
        df.loc[
            ~conflict_mask
        ]
        .reset_index(drop=True)
    )


print(
    f"\nRows removed because of conflicting labels: "
    f"{conflicting_rows_removed}"
)

print(
    f"Rows remaining after conflict removal: "
    f"{len(df)}"
)


# ============================================================
# VERIFY NO CONFLICTS REMAIN
# ============================================================

remaining_conflicts = (
    df
    .groupby(
        feature_columns,
        dropna=False,
    )["target_binary"]
    .nunique()
)

remaining_conflicts = (
    remaining_conflicts[
        remaining_conflicts > 1
    ]
)

if len(remaining_conflicts) > 0:

    raise RuntimeError(
        "Conflicting feature groups still remain "
        "after cleaning."
    )


# ============================================================
# FINAL TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("8. FINAL TARGET DISTRIBUTION")
print("=" * 80)

final_counts = (
    df["target_binary"]
    .value_counts()
    .sort_index()
)

final_percent = (
    df["target_binary"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)

for label in [0, 1]:

    label_name = (
        "No"
        if label == 0
        else "Yes"
    )

    count = int(
        final_counts.get(
            label,
            0,
        )
    )

    percent = float(
        final_percent.get(
            label,
            0,
        )
    )

    print(
        f"  {label_name}: "
        f"{count} "
        f"({percent:.2f}%)"
    )


# ============================================================
# FEATURE ENCODING
# ============================================================

print("\n" + "=" * 80)
print("9. FEATURE ENCODING")
print("=" * 80)

X_raw = df[
    feature_columns
].copy()

y = df[
    "target_binary"
].to_numpy(
    dtype=np.int64
)


# ------------------------------------------------------------
# One-hot encoding
#
# NOTE:
# This encoding is used here to inspect the resulting
# feature matrix.
#
# During actual model benchmarking, the encoder must be
# fitted only on the training data.
# ------------------------------------------------------------

try:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

except TypeError:

    # Compatibility with older scikit-learn versions.

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False,
    )


X_encoded = encoder.fit_transform(
    X_raw
)

X_encoded = np.asarray(
    X_encoded,
    dtype=np.float32,
)

encoded_feature_names = (
    encoder
    .get_feature_names_out(
        feature_columns
    )
)

print(
    f"\nOriginal feature columns: "
    f"{len(feature_columns)}"
)

print(
    f"Encoded feature columns: "
    f"{len(encoded_feature_names)}"
)

print(
    f"Encoded matrix shape: "
    f"{X_encoded.shape}"
)


# ============================================================
# GROUP IDENTIFIERS
# ============================================================

print("\n" + "=" * 80)
print("GROUP IDENTIFIERS")
print("=" * 80)

# Every identical questionnaire feature combination
# receives the same group ID.

group_index = pd.MultiIndex.from_frame(
    X_raw.reset_index(drop=True)
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

print(
    f"\nUnique groups: "
    f"{len(np.unique(group_ids))}"
)


# ============================================================
# GROUP-AWARE SPLIT
# ============================================================

print("\n" + "=" * 80)
print("10. GROUP-AWARE TRAIN / VALIDATION / TEST SPLIT")
print("=" * 80)

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

folds = list(
    splitter.split(
        X_encoded,
        y,
        groups=group_ids,
    )
)

# ------------------------------------------------------------
# Fixed folds make the experiment reproducible.
# ------------------------------------------------------------

TEST_FOLD = 4
VAL_FOLD = 3

test_indices = folds[
    TEST_FOLD
][1]

val_indices = folds[
    VAL_FOLD
][1]

test_set = set(
    test_indices.tolist()
)

val_set = set(
    val_indices.tolist()
)

train_indices = np.array(
    [
        index
        for index in range(
            len(df)
        )
        if (
            index not in test_set
            and index not in val_set
        )
    ],
    dtype=np.int64,
)


# ============================================================
# LEAKAGE CHECK
# ============================================================

train_groups = set(
    group_ids[
        train_indices
    ]
)

val_groups = set(
    group_ids[
        val_indices
    ]
)

test_groups = set(
    group_ids[
        test_indices
    ]
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
    f"\nTrain samples: "
    f"{len(train_indices)}"
)

print(
    f"Validation samples: "
    f"{len(val_indices)}"
)

print(
    f"Test samples: "
    f"{len(test_indices)}"
)

print("\nGroup leakage:")

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
    len(train_val_overlap) != 0
    or len(train_test_overlap) != 0
    or len(val_test_overlap) != 0
):

    raise RuntimeError(
        "GROUP LEAKAGE DETECTED."
    )

print(
    "\nPASS: No group leakage."
)


# ============================================================
# CLASS DISTRIBUTION FUNCTION
# ============================================================

def print_distribution(
    name,
    indices,
):

    values = y[
        indices
    ]

    counts = np.bincount(
        values,
        minlength=2,
    )

    total = len(values)

    print(
        f"\n{name}: {total}"
    )

    print(
        f"  No : "
        f"{counts[0]} "
        f"({counts[0] / total * 100:.2f}%)"
    )

    print(
        f"  Yes: "
        f"{counts[1]} "
        f"({counts[1] / total * 100:.2f}%)"
    )


# ============================================================
# CLASS DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 80)
print("11. CLASS DISTRIBUTIONS")
print("=" * 80)

print_distribution(
    "TRAIN",
    train_indices,
)

print_distribution(
    "VALIDATION",
    val_indices,
)

print_distribution(
    "TEST",
    test_indices,
)


# ============================================================
# GROUP SIZE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("12. GROUP SIZE ANALYSIS")
print("=" * 80)

final_group_sizes = (
    pd.Series(
        group_ids
    )
    .value_counts()
)

print(
    f"\nUnique groups: "
    f"{len(final_group_sizes)}"
)

print(
    f"Largest group: "
    f"{int(final_group_sizes.max())}"
)

print(
    "\nGroup size distribution:"
)

print(
    final_group_sizes
    .value_counts()
    .sort_index()
)


# ============================================================
# SPLIT ARRAYS
# ============================================================

X_train = X_encoded[
    train_indices
]

X_val = X_encoded[
    val_indices
]

X_test = X_encoded[
    test_indices
]

y_train = y[
    train_indices
]

y_val = y[
    val_indices
]

y_test = y[
    test_indices
]

groups_train = group_ids[
    train_indices
]

groups_val = group_ids[
    val_indices
]

groups_test = group_ids[
    test_indices
]


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("13. FINAL CLEAN DATASET SUMMARY")
print("=" * 80)

print(
    f"""
Original rows:
    {original_row_count}

Rows removed for missing values:
    {removed_missing_rows}

Rows removed as exact duplicates:
    {duplicate_rows_removed}

Rows removed as unknown target:
    {unknown_target_count}

Rows removed as conflicting groups:
    {conflicting_rows_removed}

Final clean rows:
    {len(df)}

Final feature columns before encoding:
    {len(feature_columns)}

Final feature columns after encoding:
    {X_encoded.shape[1]}

Training samples:
    {len(X_train)}

Validation samples:
    {len(X_val)}

Test samples:
    {len(X_test)}

Unique feature groups:
    {len(np.unique(group_ids))}
"""
)


# ============================================================
# FINAL INTEGRITY CHECKS
# ============================================================

print("=" * 80)
print("14. INTEGRITY CHECKS")
print("=" * 80)


checks = {
    "No missing values": (
        not df.isna().any().any()
    ),

    "No unknown targets": (
        UNKNOWN_TARGET
        not in df[TARGET].values
    ),

    "No conflicting groups": (
        len(remaining_conflicts) == 0
    ),

    "No train/validation group leakage": (
        len(train_val_overlap) == 0
    ),

    "No train/test group leakage": (
        len(train_test_overlap) == 0
    ),

    "No validation/test group leakage": (
        len(val_test_overlap) == 0
    ),

    "Binary target only": (
        set(
            np.unique(y)
        ).issubset({0, 1})
    ),

    "Finite encoded features": (
        np.isfinite(
            X_encoded
        ).all()
    ),
}


all_passed = True

for check_name, passed in checks.items():

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"{status} - {check_name}"
    )

    if not passed:
        all_passed = False


print("\n" + "=" * 80)

if not all_passed:

    print(
        "ONE OR MORE CHECKS FAILED."
    )

    raise RuntimeError(
        "Data integrity checks failed."
    )


print(
    "ALL DATA CLEANING AND SPLIT CHECKS PASSED."
)

print("=" * 80)

print(
    "\nNo model was trained."
)

print(
    "No model file was saved."
)

print(
    "\nPOSTNATAL DATA PREPARATION COMPLETE."
)

print("=" * 80)