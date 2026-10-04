"""
SERENOVA — CTU-UHB CTG PREPROCESSING

Purpose
-------
Convert the raw CTU-UHB intrapartum cardiotocography WFDB
records into the standard Serenova processed hierarchy.

Raw:
    data/raw/ctu-chb/

Processed:
    data/processed/ctu_uhb_ctg/

Signals:
    FHR
    UC

The script:
    1. Discovers WFDB records.
    2. Reads headers and signals.
    3. Identifies FHR and UC channels.
    4. Validates finite values.
    5. Performs record-level train/val/test splitting.
    6. Creates fixed 1200-sample windows.
    7. Rejects windows with excessive invalid values.
    8. Fits normalization ONLY on training data.
    9. Applies the same normalization to val/test.
   10. Saves X.npy, groups.npy and metadata.json.

IMPORTANT
---------
This is waveform representation preprocessing.
No clinical label is invented.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import wfdb

from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "ctu-chb"
)

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "ctu_uhb_ctg"
)

INTERIM_DIR = (
    BASE_DIR
    / "data"
    / "interim"
    / "ctu_uhb_ctg"
)

QUALITY_DIR = (
    INTERIM_DIR
    / "quality"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

WINDOW_SIZE = 1200

STRIDE = 1200

# Maximum fraction of non-finite / invalid values allowed.
MAX_INVALID_FRACTION = 0.05

# Maximum fraction of samples that can be clipped to
# physiologically implausible signal limits.
MAX_CLIPPED_FRACTION = 0.05


# CTG signals are normally low-frequency.
# We use conservative filtering and preserve the processed
# temporal resolution rather than resampling arbitrarily.
FILTER_ORDER = 4

# Expected CTG channel ranges used only for quality control.
# These are NOT diagnostic labels.
FHR_MIN = 30.0
FHR_MAX = 240.0

UC_MIN = -20.0
UC_MAX = 300.0


# ============================================================
# DIRECTORIES
# ============================================================

for directory in [
    PROCESSED_DIR,
    INTERIM_DIR,
    QUALITY_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# RECORD DISCOVERY
# ============================================================

def discover_records():

    records = []

    for hea_path in RAW_DIR.rglob("*.hea"):

        record_path = hea_path.with_suffix("")

        records.append(
            record_path
        )

    records = sorted(
        records,
        key=lambda p: str(p).lower(),
    )

    if not records:

        raise RuntimeError(
            f"No WFDB .hea records found under:\n{RAW_DIR}"
        )

    return records


# ============================================================
# CHANNEL IDENTIFICATION
# ============================================================

def normalize_channel_name(name: str) -> str:

    return (
        name
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def find_channel(
    signal_names,
    candidates,
):

    normalized = {
        normalize_channel_name(name): index
        for index, name
        in enumerate(signal_names)
    }

    for candidate in candidates:

        candidate_normalized = normalize_channel_name(
            candidate
        )

        if candidate_normalized in normalized:

            return normalized[
                candidate_normalized
            ]

    # Partial matching.
    for index, name in enumerate(signal_names):

        normalized_name = normalize_channel_name(
            name
        )

        for candidate in candidates:

            candidate_normalized = (
                normalize_channel_name(
                    candidate
                )
            )

            if (
                candidate_normalized
                in normalized_name
            ):

                return index

    return None


def identify_channels(signal_names):

    fhr_candidates = [
        "FHR",
        "FHR1",
        "FHR2",
        "FHR_M",
        "FetalHeartRate",
        "Fetal Heart Rate",
    ]

    uc_candidates = [
        "UC",
        "TOCO",
        "Toco",
        "UterineContraction",
        "Uterine Contractions",
    ]

    fhr_index = find_channel(
        signal_names,
        fhr_candidates,
    )

    uc_index = find_channel(
        signal_names,
        uc_candidates,
    )

    return fhr_index, uc_index


# ============================================================
# QUALITY FILTERING
# ============================================================

def interpolate_invalid(
    signal: np.ndarray,
) -> np.ndarray:

    signal = signal.astype(
        np.float32,
        copy=True,
    )

    finite = np.isfinite(signal)

    if finite.all():
        return signal

    indices = np.arange(
        len(signal)
    )

    if finite.sum() < 2:

        return np.full_like(
            signal,
            np.nan,
        )

    signal[~finite] = np.interp(
        indices[~finite],
        indices[finite],
        signal[finite],
    )

    return signal


def bandpass_filter(
    signal,
    fs,
    low_hz,
    high_hz,
):

    nyquist = fs / 2.0

    if (
        low_hz <= 0
        or high_hz >= nyquist
        or low_hz >= high_hz
    ):

        return signal

    b, a = butter(
        FILTER_ORDER,
        [
            low_hz / nyquist,
            high_hz / nyquist,
        ],
        btype="band",
    )

    pad_length = 3 * max(
        len(a),
        len(b),
    )

    if len(signal) <= pad_length:

        return signal

    return filtfilt(
        b,
        a,
        signal,
    ).astype(
        np.float32
    )


def clean_signal(
    signal,
    minimum,
    maximum,
):

    signal = signal.astype(
        np.float32,
        copy=True,
    )

    invalid = ~np.isfinite(signal)

    invalid_fraction = (
        float(invalid.mean())
    )

    signal = interpolate_invalid(
        signal
    )

    if not np.isfinite(signal).all():

        return None, {
            "invalid_fraction":
                invalid_fraction,
            "clipped_fraction":
                1.0,
            "accepted":
                False,
        }

    clipped = (
        (signal < minimum)
        |
        (signal > maximum)
    )

    clipped_fraction = (
        float(clipped.mean())
    )

    signal = np.clip(
        signal,
        minimum,
        maximum,
    )

    accepted = (
        invalid_fraction
        <= MAX_INVALID_FRACTION
        and
        clipped_fraction
        <= MAX_CLIPPED_FRACTION
    )

    return signal, {
        "invalid_fraction":
            invalid_fraction,
        "clipped_fraction":
            clipped_fraction,
        "accepted":
            bool(accepted),
    }


# ============================================================
# RECORD PROCESSING
# ============================================================

def process_record(
    record_path: Path,
):

    record_name = record_path.name

    try:

        record = wfdb.rdrecord(
            str(record_path)
        )

    except Exception as exc:

        return None, {
            "record":
                record_name,
            "status":
                "read_error",
            "error":
                str(exc),
        }

    signal_names = (
        record.sig_name
    )

    fhr_index, uc_index = (
        identify_channels(
            signal_names
        )
    )

    if (
        fhr_index is None
        or uc_index is None
    ):

        return None, {
            "record":
                record_name,
            "status":
                "missing_required_channels",
            "signal_names":
                signal_names,
            "fhr_index":
                fhr_index,
            "uc_index":
                uc_index,
        }

    fs = float(
        record.fs
    )

    signals = record.p_signal

    if signals is None:

        return None, {
            "record":
                record_name,
            "status":
                "no_signal",
        }

    fhr = signals[
        :,
        fhr_index
    ]

    uc = signals[
        :,
        uc_index
    ]

    fhr, fhr_quality = clean_signal(
        fhr,
        FHR_MIN,
        FHR_MAX,
    )

    uc, uc_quality = clean_signal(
        uc,
        UC_MIN,
        UC_MAX,
    )

    if (
        fhr is None
        or uc is None
        or not fhr_quality["accepted"]
        or not uc_quality["accepted"]
    ):

        return None, {
            "record":
                record_name,
            "status":
                "quality_rejected",
            "sampling_frequency":
                fs,
            "signal_names":
                signal_names,
            "fhr_quality":
                fhr_quality,
            "uc_quality":
                uc_quality,
        }

    # --------------------------------------------------------
    # Filtering
    #
    # FHR and UC are low-frequency CTG signals.
    # Only apply filtering when the native sampling frequency
    # supports the requested range.
    # --------------------------------------------------------

    fhr = bandpass_filter(
        fhr,
        fs,
        0.01,
        min(1.5, fs / 2.0 * 0.95),
    )

    uc = bandpass_filter(
        uc,
        fs,
        0.005,
        min(0.7, fs / 2.0 * 0.95),
    )

    signals_out = np.stack(
        [
            fhr,
            uc,
        ],
        axis=0,
    ).astype(
        np.float32
    )

    return signals_out, {
        "record":
            record_name,
        "status":
            "accepted",
        "sampling_frequency":
            fs,
        "signal_names":
            signal_names,
        "selected_channels": {
            "channel_0":
                signal_names[fhr_index],
            "channel_1":
                signal_names[uc_index],
        },
        "num_samples":
            int(signals_out.shape[1]),
        "duration_seconds":
            float(
                signals_out.shape[1]
                / fs
            ),
        "fhr_quality":
            fhr_quality,
        "uc_quality":
            uc_quality,
    }


# ============================================================
# WINDOWING
# ============================================================

def make_windows(
    signal,
    record_id,
):

    windows = []
    groups = []

    total_samples = signal.shape[1]

    start = 0

    while (
        start + WINDOW_SIZE
        <= total_samples
    ):

        window = signal[
            :,
            start:
            start + WINDOW_SIZE,
        ]

        if np.isfinite(
            window
        ).all():

            windows.append(
                window.astype(
                    np.float32
                )
            )

            groups.append(
                record_id
            )

        start += STRIDE

    return windows, groups


# ============================================================
# RECORD-LEVEL SPLIT
# ============================================================

def split_records(
    records,
):

    records = list(records)

    train_records, temp_records = (
        train_test_split(
            records,
            test_size=0.30,
            random_state=SEED,
        )
    )

    val_records, test_records = (
        train_test_split(
            temp_records,
            test_size=0.50,
            random_state=SEED,
        )
    )

    return (
        sorted(train_records),
        sorted(val_records),
        sorted(test_records),
    )


# ============================================================
# TRAIN-ONLY NORMALIZATION
# ============================================================

def fit_normalization(
    train_x,
):

    # Per-channel global normalization.
    mean = train_x.mean(
        axis=(0, 2)
    )

    std = train_x.std(
        axis=(0, 2)
    )

    std = np.maximum(
        std,
        1e-6,
    )

    return (
        mean.astype(
            np.float32
        ),
        std.astype(
            np.float32
        ),
    )


def apply_normalization(
    x,
    mean,
    std,
):

    return (
        (
            x
            -
            mean[None, :, None]
        )
        /
        std[None, :, None]
    ).astype(
        np.float32
    )


# ============================================================
# SAVE SPLIT
# ============================================================

def save_split(
    name,
    x,
    groups,
):

    split_dir = (
        PROCESSED_DIR
        / name
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        split_dir / "X.npy",
        x.astype(
            np.float32
        ),
    )

    np.save(
        split_dir / "groups.npy",
        np.asarray(
            groups
        ),
    )

    metadata = {
        "dataset":
            "ctu_uhb_ctg",
        "split":
            name,
        "shape":
            list(x.shape),
        "dtype":
            str(x.dtype),
        "window_size_samples":
            WINDOW_SIZE,
        "stride_samples":
            STRIDE,
        "normalization":
            "train_only_global_channel_zscore",
        "groups":
            "record_ids",
    }

    with open(
        split_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("SERENOVA — CTU-UHB CTG PREPROCESSING")
    print("=" * 72)

    print()
    print(
        f"Raw directory:\n{RAW_DIR}"
    )

    print()
    print(
        f"Processed directory:\n{PROCESSED_DIR}"
    )

    # --------------------------------------------------------
    # DISCOVER
    # --------------------------------------------------------

    records = discover_records()

    print()
    print(
        f"Discovered WFDB records: {len(records)}"
    )

    # --------------------------------------------------------
    # PROCESS ALL RECORDS
    # --------------------------------------------------------

    accepted = {}

    report = []

    print()
    print("=" * 72)
    print("READING RECORDS")
    print("=" * 72)

    for index, record_path in enumerate(
        records,
        start=1,
    ):

        signal, metadata = process_record(
            record_path
        )

        report.append(
            metadata
        )

        if signal is not None:

            accepted[
                metadata["record"]
            ] = (
                signal,
                metadata,
            )

        print(
            f"[{index:03d}/{len(records):03d}] "
            f"{metadata.get('record')} "
            f"-> "
            f"{metadata.get('status')}"
        )

    print()
    print(
        f"Accepted records: {len(accepted)}"
    )

    if not accepted:

        raise RuntimeError(
            "No CTU records passed preprocessing."
        )

    # --------------------------------------------------------
    # SPLIT ACCEPTED RECORDS
    # --------------------------------------------------------

    accepted_ids = sorted(
        accepted.keys()
    )

    train_ids, val_ids, test_ids = (
        split_records(
            accepted_ids
        )
    )

    print()
    print("=" * 72)
    print("RECORD-LEVEL SPLIT")
    print("=" * 72)

    print(
        f"Train records: {len(train_ids)}"
    )

    print(
        f"Val records:   {len(val_ids)}"
    )

    print(
        f"Test records:  {len(test_ids)}"
    )

    # --------------------------------------------------------
    # WINDOW RECORDS
    # --------------------------------------------------------

    def build_split(
        record_ids
    ):

        all_windows = []
        all_groups = []

        for record_id in record_ids:

            signal = accepted[
                record_id
            ][0]

            windows, groups = (
                make_windows(
                    signal,
                    record_id,
                )
            )

            all_windows.extend(
                windows
            )

            all_groups.extend(
                groups
            )

        if not all_windows:

            raise RuntimeError(
                "No windows generated."
            )

        return (
            np.stack(
                all_windows
            ).astype(
                np.float32
            ),
            np.asarray(
                all_groups
            ),
        )

    X_train, groups_train = (
        build_split(
            train_ids
        )
    )

    X_val, groups_val = (
        build_split(
            val_ids
        )
    )

    X_test, groups_test = (
        build_split(
            test_ids
        )
    )

    print()
    print("=" * 72)
    print("RAW WINDOW COUNTS")
    print("=" * 72)

    print(
        f"Train: {X_train.shape}"
    )

    print(
        f"Val:   {X_val.shape}"
    )

    print(
        f"Test:  {X_test.shape}"
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    print()
    print(
        "Fitting normalization on TRAIN ONLY..."
    )

    train_mean, train_std = (
        fit_normalization(
            X_train
        )
    )

    X_train = apply_normalization(
        X_train,
        train_mean,
        train_std,
    )

    X_val = apply_normalization(
        X_val,
        train_mean,
        train_std,
    )

    X_test = apply_normalization(
        X_test,
        train_mean,
        train_std,
    )

    # --------------------------------------------------------
    # FINITE CHECK
    # --------------------------------------------------------

    for name, array in [
        ("train", X_train),
        ("val", X_val),
        ("test", X_test),
    ]:

        if not np.isfinite(
            array
        ).all():

            raise RuntimeError(
                f"Non-finite values after "
                f"normalization: {name}"
            )

    # --------------------------------------------------------
    # SAVE SPLITS
    # --------------------------------------------------------

    save_split(
        "train",
        X_train,
        groups_train,
    )

    save_split(
        "val",
        X_val,
        groups_val,
    )

    save_split(
        "test",
        X_test,
        groups_test,
    )

    # --------------------------------------------------------
    # VERIFY NO GROUP LEAKAGE
    # --------------------------------------------------------

    train_set = set(
        groups_train.tolist()
    )

    val_set = set(
        groups_val.tolist()
    )

    test_set = set(
        groups_test.tolist()
    )

    overlap_train_val = (
        train_set & val_set
    )

    overlap_train_test = (
        train_set & test_set
    )

    overlap_val_test = (
        val_set & test_set
    )

    if (
        overlap_train_val
        or overlap_train_test
        or overlap_val_test
    ):

        raise RuntimeError(
            "Record leakage detected."
        )

    # --------------------------------------------------------
    # SAVE NORMALIZATION
    # --------------------------------------------------------

    normalization = {
        "method":
            "train_only_global_channel_zscore",

        "fit_split":
            "train",

        "channel_order": [
            "FHR",
            "UC",
        ],

        "mean": [
            float(x)
            for x in train_mean
        ],

        "std": [
            float(x)
            for x in train_std
        ],
    }

    with open(
        PROCESSED_DIR
        / "normalization.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            normalization,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # DATASET METADATA
    # --------------------------------------------------------

    dataset_metadata = {

        "dataset":
            "ctu_uhb_ctg",

        "seed":
            SEED,

        "raw_directory":
            str(RAW_DIR),

        "record_count_discovered":
            len(records),

        "record_count_accepted":
            len(accepted),

        "train_records":
            len(train_ids),

        "val_records":
            len(val_ids),

        "test_records":
            len(test_ids),

        "train_windows":
            int(X_train.shape[0]),

        "val_windows":
            int(X_val.shape[0]),

        "test_windows":
            int(X_test.shape[0]),

        "input_shape":
            [2, WINDOW_SIZE],

        "window_size_samples":
            WINDOW_SIZE,

        "stride_samples":
            STRIDE,

        "channel_order": [
            "FHR",
            "UC",
        ],

        "normalization":
            "train_only_global_channel_zscore",

        "group_split":
            "record_level",

        "group_overlap": {
            "train_val":
                list(overlap_train_val),
            "train_test":
                list(overlap_train_test),
            "val_test":
                list(overlap_val_test),
        },
    }

    with open(
        PROCESSED_DIR
        / "dataset_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            dataset_metadata,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # INTERIM REPORT
    # --------------------------------------------------------

    validation_report = {

        "dataset":
            "ctu_uhb_ctg",

        "records_discovered":
            len(records),

        "records_accepted":
            len(accepted),

        "records_rejected":
            len(records)
            - len(accepted),

        "train_records":
            len(train_ids),

        "val_records":
            len(val_ids),

        "test_records":
            len(test_ids),

        "train_windows":
            int(X_train.shape[0]),

        "val_windows":
            int(X_val.shape[0]),

        "test_windows":
            int(X_test.shape[0]),

        "train_val_overlap":
            list(overlap_train_val),

        "train_test_overlap":
            list(overlap_train_test),

        "val_test_overlap":
            list(overlap_val_test),
    }

    with open(
        INTERIM_DIR
        / "validation_report.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            validation_report,
            f,
            indent=4,
        )

    with open(
        INTERIM_DIR
        / "record_index.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("CTU-UHB CTG PREPROCESSING COMPLETE")
    print("=" * 72)

    print()
    print(
        f"Processed directory:\n{PROCESSED_DIR}"
    )

    print()
    print(
        f"Train: {X_train.shape}"
    )

    print(
        f"Val:   {X_val.shape}"
    )

    print(
        f"Test:  {X_test.shape}"
    )

    print()
    print(
        "Record leakage:"
    )

    print(
        f"Train/Val : {overlap_train_val}"
    )

    print(
        f"Train/Test: {overlap_train_test}"
    )

    print(
        f"Val/Test  : {overlap_val_test}"
    )


if __name__ == "__main__":
    main()