from pathlib import Path
import json
import csv
import numpy as np
from scipy import stats
from scipy.signal import butter, filtfilt, welch
from sklearn.model_selection import train_test_split
import wfdb


ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "bidmc"
    / "bidmc-ppg-and-respiration-dataset-1.0.0"
)
if not RAW_DIR.exists():
    RAW_DIR = (
        ROOT
        / "data"
        / "raw"
        / "external"
        / "bidmc"
        / "bidmc-ppg-and-respiration-dataset-1.0.0"
    )

INTERIM_DIR = ROOT / "data" / "interim" / "bidmc"
QUALITY_DIR = INTERIM_DIR / "quality"
PROCESSED_DIR = ROOT / "data" / "processed" / "bidmc"
RESULTS_DIR = ROOT / "results" / "bidmc"

# Create required directories
for d in [INTERIM_DIR, QUALITY_DIR, PROCESSED_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    (PROCESSED_DIR / split).mkdir(parents=True, exist_ok=True)

(RESULTS_DIR / "model").mkdir(parents=True, exist_ok=True)


FS = 125
WINDOW_SECONDS = 10
WINDOW_SIZE = FS * WINDOW_SECONDS  # 1250 samples


def bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float, fs: int = FS, order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def zscore_normalize(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize signal using per-window z-score."""
    mean = np.mean(signal)
    std = np.std(signal)
    if std < eps:
        return signal - mean
    return (signal - mean) / std


def find_records() -> list[str]:
    """Find all standard signal record stems (excluding 'n' numerics)."""
    records = []
    for hea in RAW_DIR.glob("*.hea"):
        name = hea.stem
        if not name.endswith("n"):
            records.append(name)
    return sorted(set(records))


def compute_sqi(ppg_raw: np.ndarray, resp_raw: np.ndarray, fs: int = FS) -> dict:
    """Compute Signal Quality Index (SQI) metrics for raw signal channels."""
    ppg_raw = np.asarray(ppg_raw, dtype=np.float64)
    resp_raw = np.asarray(resp_raw, dtype=np.float64)
    
    # PPG Skewness and Kurtosis
    ppg_skew = float(stats.skew(ppg_raw))
    ppg_kurt = float(stats.kurtosis(ppg_raw))
    
    # Perfusion Index approximation (AC / DC * 100)
    ppg_dc = float(np.mean(ppg_raw))
    ppg_ac = float(np.std(ppg_raw))
    perfusion_index = float((ppg_ac / (abs(ppg_dc) + 1e-6)) * 100.0)

    # Power Spectral Density (PSD) relative power in physiological cardiac band (0.5 - 5.0 Hz)
    freqs, psd = welch(ppg_raw, fs=fs, nperseg=min(len(ppg_raw), 512))
    cardiac_band = (freqs >= 0.5) & (freqs <= 5.0)
    total_power = np.sum(psd) + 1e-12
    cardiac_power = np.sum(psd[cardiac_band])
    ppg_power_sqi = float(cardiac_power / total_power)

    # Respiration power in physiological respiratory band (0.1 - 0.7 Hz)
    rfreqs, rpsd = welch(resp_raw, fs=fs, nperseg=min(len(resp_raw), 512))
    resp_band = (rfreqs >= 0.1) & (rfreqs <= 0.7)
    rtotal_power = np.sum(rpsd) + 1e-12
    rband_power = np.sum(rpsd[resp_band])
    resp_power_sqi = float(rband_power / rtotal_power)

    # Zero crossing rate on demeaned signals
    ppg_centered = ppg_raw - np.mean(ppg_raw)
    ppg_zcr = float(np.mean(np.diff(ppg_centered > 0) != 0))

    # Overall Quality Score (0 to 1 composite metric)
    quality_score = float(np.clip(
        0.4 * ppg_power_sqi + 0.3 * resp_power_sqi + 0.3 * (1.0 / (1.0 + abs(ppg_skew))),
        0.0,
        1.0
    ))

    return {
        "ppg_skewness": round(ppg_skew, 4),
        "ppg_kurtosis": round(ppg_kurt, 4),
        "perfusion_index_pct": round(perfusion_index, 4),
        "ppg_cardiac_band_power_sqi": round(ppg_power_sqi, 4),
        "resp_band_power_sqi": round(resp_power_sqi, 4),
        "ppg_zero_crossing_rate": round(ppg_zcr, 4),
        "composite_quality_score": round(quality_score, 4)
    }


def validate_and_index_records(records: list[str]) -> tuple[dict, list[dict], dict]:
    """Validate all raw records, produce record index and quality reports."""
    record_index = []
    validation_issues = []
    quality_metrics = {}

    for record_name in records:
        rec_path = str(RAW_DIR / record_name)
        breath_path = RAW_DIR / f"{record_name}.breath"
        has_breath = breath_path.exists()

        try:
            record = wfdb.rdrecord(rec_path)
            sig_names = [n.strip().strip(",").upper() for n in record.sig_name]

            has_pleth = "PLETH" in sig_names
            has_resp = "RESP" in sig_names

            if not has_pleth or not has_resp:
                validation_issues.append({
                    "record": record_name,
                    "issue": f"Missing required channels. Present: {sig_names}"
                })

            pleth_idx = sig_names.index("PLETH") if has_pleth else -1
            resp_idx = sig_names.index("RESP") if has_resp else -1

            sig_len = int(record.sig_len)
            fs = int(record.fs)
            duration_s = float(sig_len / fs)
            expected_windows = int(sig_len // WINDOW_SIZE)

            # Signal quality
            if has_pleth and has_resp:
                raw_ppg = record.p_signal[:, pleth_idx]
                raw_resp = record.p_signal[:, resp_idx]
                sqi = compute_sqi(raw_ppg, raw_resp, fs=fs)
            else:
                sqi = {"composite_quality_score": 0.0}

            quality_metrics[record_name] = sqi

            record_index.append({
                "record_name": record_name,
                "sampling_frequency_hz": fs,
                "signal_length": sig_len,
                "duration_seconds": round(duration_s, 2),
                "total_channels": len(sig_names),
                "channel_names": ";".join(record.sig_name),
                "has_pleth": has_pleth,
                "has_resp": has_resp,
                "has_breath_annotation": has_breath,
                "expected_windows": expected_windows,
                "quality_score": sqi.get("composite_quality_score", 0.0)
            })

        except Exception as e:
            validation_issues.append({
                "record": record_name,
                "issue": str(e)
            })

    validation_report = {
        "dataset_name": "BIDMC PPG and Respiration Dataset",
        "dataset_version": "1.0.0",
        "raw_directory": str(RAW_DIR),
        "total_records_discovered": len(records),
        "valid_records_count": len(record_index) - len(validation_issues),
        "validation_passed": len(validation_issues) == 0,
        "validation_issues": validation_issues,
        "sampling_frequency_hz": FS,
        "window_size_samples": WINDOW_SIZE,
        "channels_required": ["PLETH", "RESP"]
    }

    return validation_report, record_index, quality_metrics


def process_record(record_name: str) -> list[np.ndarray]:
    """Process a single record: load, filter, normalize, and extract windows."""
    record_path = str(RAW_DIR / record_name)
    record = wfdb.rdrecord(record_path)

    signal_names = [name.strip().strip(",").upper() for name in record.sig_name]

    if "PLETH" not in signal_names:
        raise ValueError(f"PLETH channel not found in {record_name}")
    if "RESP" not in signal_names:
        raise ValueError(f"RESP channel not found in {record_name}")

    ppg_index = signal_names.index("PLETH")
    resp_index = signal_names.index("RESP")

    ppg = record.p_signal[:, ppg_index]
    resp = record.p_signal[:, resp_index]

    # Physiological bandpass filtering
    # PPG: 0.5 - 5.0 Hz
    ppg_filtered = bandpass_filter(ppg, lowcut=0.5, highcut=5.0, fs=FS, order=4)
    # Respiration: 0.1 - 0.7 Hz
    resp_filtered = bandpass_filter(resp, lowcut=0.1, highcut=0.7, fs=FS, order=4)

    # Per-window z-score normalization
    windows = []
    number_of_windows = len(ppg_filtered) // WINDOW_SIZE

    for i in range(number_of_windows):
        start = i * WINDOW_SIZE
        end = start + WINDOW_SIZE

        ppg_w = ppg_filtered[start:end]
        resp_w = resp_filtered[start:end]

        if len(ppg_w) != WINDOW_SIZE or len(resp_w) != WINDOW_SIZE:
            continue

        if not (np.isfinite(ppg_w).all() and np.isfinite(resp_w).all()):
            continue

        ppg_norm = zscore_normalize(ppg_w)
        resp_norm = zscore_normalize(resp_w)

        # Shape: [2, WINDOW_SIZE] (channel 0: PPG, channel 1: Respiration)
        window = np.stack([ppg_norm, resp_norm], axis=0).astype(np.float32)
        windows.append(window)

    return windows


def main():
    print("\n" + "=" * 70)
    print("SERENOVA — BIDMC PPG & RESPIRATION WAVEFORM PREPROCESSING")
    print("=" * 70)

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw BIDMC directory not found:\n{RAW_DIR}")

    records = find_records()
    print(f"Discovered records: {len(records)}")

    if not records:
        raise RuntimeError("No BIDMC records found.")

    # -----------------------------------------------------------------
    # 1. Validation & Interim Reports
    # -----------------------------------------------------------------
    print("\n[1/4] Running validation and computing signal quality indices (SQI)...")
    val_report, rec_index, quality_metrics = validate_and_index_records(records)

    with open(INTERIM_DIR / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=4)

    # Write record_index.csv
    if rec_index:
        with open(INTERIM_DIR / "record_index.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rec_index[0].keys()))
            writer.writeheader()
            writer.writerows(rec_index)

    # Write quality metrics
    with open(QUALITY_DIR / "sqi_metrics.json", "w", encoding="utf-8") as f:
        json.dump(quality_metrics, f, indent=4)

    with open(QUALITY_DIR / "quality_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["record_name", "quality_score", "ppg_cardiac_band_power_sqi", "resp_band_power_sqi", "perfusion_index_pct", "ppg_skewness", "ppg_kurtosis"])
        for r_name, metrics in quality_metrics.items():
            writer.writerow([
                r_name,
                metrics.get("composite_quality_score"),
                metrics.get("ppg_cardiac_band_power_sqi"),
                metrics.get("resp_band_power_sqi"),
                metrics.get("perfusion_index_pct"),
                metrics.get("ppg_skewness"),
                metrics.get("ppg_kurtosis")
            ])

    print("  -> Interim validation report and SQI index generated.")

    # -----------------------------------------------------------------
    # 2. Extract and Window Waveforms
    # -----------------------------------------------------------------
    print("\n[2/4] Processing records into 10s physiological windows (1250 samples, 2 channels)...")
    all_windows = []
    record_ids = []

    for record_name in records:
        windows = process_record(record_name)
        all_windows.extend(windows)
        record_ids.extend([record_name] * len(windows))

    X = np.asarray(all_windows, dtype=np.float32)
    record_ids = np.asarray(record_ids)

    print(f"  Total extracted windows: {len(X)} (Shape: {X.shape})")

    if len(X) == 0:
        raise RuntimeError("No valid BIDMC windows generated.")

    # -----------------------------------------------------------------
    # 3. Patient/Record-Level Split
    # -----------------------------------------------------------------
    print("\n[3/4] Performing locked patient-level train / val / test split...")
    unique_records = np.unique(record_ids)

    train_records, temp_records = train_test_split(
        unique_records,
        test_size=0.30,
        random_state=42
    )

    val_records, test_records = train_test_split(
        temp_records,
        test_size=0.50,
        random_state=42
    )

    train_mask = np.isin(record_ids, train_records)
    val_mask = np.isin(record_ids, val_records)
    test_mask = np.isin(record_ids, test_records)

    X_train, groups_train = X[train_mask], record_ids[train_mask]
    X_val, groups_val = X[val_mask], record_ids[val_mask]
    X_test, groups_test = X[test_mask], record_ids[test_mask]

    print(f"  Train : {X_train.shape} across {len(train_records)} records")
    print(f"  Val   : {X_val.shape} across {len(val_records)} records")
    print(f"  Test  : {X_test.shape} across {len(test_records)} records")

    # Verify exact target shapes
    assert X_train.shape == (1776, 2, 1250), f"Expected train shape (1776, 2, 1250), got {X_train.shape}"
    assert X_val.shape == (384, 2, 1250), f"Expected val shape (384, 2, 1250), got {X_val.shape}"
    assert X_test.shape == (384, 2, 1250), f"Expected test shape (384, 2, 1250), got {X_test.shape}"

    # -----------------------------------------------------------------
    # 4. Save Standardized Processed Hierarchy
    # -----------------------------------------------------------------
    print("\n[4/4] Saving processed artifacts to disk...")

    # Save nested split folders: processed/bidmc/train, val, test
    splits_data = {
        "train": (X_train, groups_train, train_records),
        "val": (X_val, groups_val, val_records),
        "test": (X_test, groups_test, test_records),
    }

    for s_name, (s_X, s_groups, s_rec_list) in splits_data.items():
        s_dir = PROCESSED_DIR / s_name
        np.save(s_dir / "X.npy", s_X)
        np.save(s_dir / "groups.npy", s_groups)
        
        split_meta = {
            "split": s_name,
            "num_samples": int(len(s_X)),
            "shape": list(s_X.shape),
            "dtype": str(s_X.dtype),
            "records_count": len(s_rec_list),
            "records": s_rec_list.tolist() if hasattr(s_rec_list, "tolist") else list(s_rec_list),
            "channels": ["PLETH_PPG", "RESP"],
            "sampling_frequency_hz": FS,
            "window_seconds": WINDOW_SECONDS,
            "window_samples": WINDOW_SIZE
        }
        with open(s_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(split_meta, f, indent=4)

    # Backwards compatibility flat files
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_val.npy", X_val)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "groups_train.npy", groups_train)
    np.save(PROCESSED_DIR / "groups_val.npy", groups_val)
    np.save(PROCESSED_DIR / "groups_test.npy", groups_test)

    # Normalization specification
    normalization_config = {
        "method": "per-window z-score",
        "epsilon": 1e-8,
        "channels": {
            "PLETH_PPG": {
                "channel_index": 0,
                "filter_type": "butterworth_bandpass",
                "order": 4,
                "lowcut_hz": 0.5,
                "highcut_hz": 5.0,
                "sampling_frequency_hz": FS,
                "target_unit": "normalized_amplitude"
            },
            "RESP": {
                "channel_index": 1,
                "filter_type": "butterworth_bandpass",
                "order": 4,
                "lowcut_hz": 0.1,
                "highcut_hz": 0.7,
                "sampling_frequency_hz": FS,
                "target_unit": "normalized_amplitude"
            }
        }
    }
    with open(PROCESSED_DIR / "normalization.json", "w", encoding="utf-8") as f:
        json.dump(normalization_config, f, indent=4)

    # Dataset overall metadata
    dataset_metadata = {
        "dataset_name": "BIDMC PPG and Respiration Dataset",
        "dataset_version": "1.0.0",
        "source": "PhysioNet / Beth Israel Deaconess Medical Center",
        "sampling_frequency_hz": FS,
        "window_seconds": WINDOW_SECONDS,
        "window_samples": WINDOW_SIZE,
        "channels": ["PLETH_PPG", "RESP"],
        "channel_order": [0, 1],
        "total_records": len(unique_records),
        "total_windows": int(len(X)),
        "split_strategy": "record-level locked split (random_state=42, test_size=0.30, val_size=0.15, test_size=0.15)",
        "train_records": train_records.tolist(),
        "val_records": val_records.tolist(),
        "test_records": test_records.tolist(),
        "shapes": {
            "train": list(X_train.shape),
            "val": list(X_val.shape),
            "test": list(X_test.shape)
        },
        "normalization": normalization_config
    }
    with open(PROCESSED_DIR / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(dataset_metadata, f, indent=4)

    # Preprocessing Report in results/bidmc
    preprocessing_report = {
        "dataset": "BIDMC PPG and Respiration",
        "status": "COMPLETED_SUCCESSFULLY",
        "raw_records_processed": len(unique_records),
        "total_windows_generated": int(len(X)),
        "windows_per_record": 48,
        "sampling_rate_hz": FS,
        "window_duration_seconds": WINDOW_SECONDS,
        "window_samples": WINDOW_SIZE,
        "channels": {
            "channel_0": "PLETH_PPG (0.5-5.0 Hz bandpass)",
            "channel_1": "RESP (0.1-0.7 Hz bandpass)"
        },
        "splits": {
            "train": {
                "records_count": len(train_records),
                "windows_count": int(len(X_train)),
                "shape": list(X_train.shape)
            },
            "val": {
                "records_count": len(val_records),
                "windows_count": int(len(X_val)),
                "shape": list(X_val.shape)
            },
            "test": {
                "records_count": len(test_records),
                "windows_count": int(len(X_test)),
                "shape": list(X_test.shape)
            }
        },
        "quality_summary": {
            "mean_composite_quality_score": round(float(np.mean([m["composite_quality_score"] for m in quality_metrics.values()])), 4),
            "min_composite_quality_score": round(float(np.min([m["composite_quality_score"] for m in quality_metrics.values()])), 4),
            "max_composite_quality_score": round(float(np.max([m["composite_quality_score"] for m in quality_metrics.values()])), 4)
        }
    }
    with open(RESULTS_DIR / "preprocessing_report.json", "w", encoding="utf-8") as f:
        json.dump(preprocessing_report, f, indent=4)

    print("\n" + "=" * 70)
    print("BIDMC PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Interim Data   : {INTERIM_DIR}")
    print(f"Processed Data : {PROCESSED_DIR}")
    print(f"Results Report : {RESULTS_DIR / 'preprocessing_report.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()