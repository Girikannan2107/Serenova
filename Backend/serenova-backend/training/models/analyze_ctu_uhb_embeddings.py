from pathlib import Path
import json
import numpy as np


# ============================================================
# CTU-UHB CTG EMBEDDING DIAGNOSTICS
# ============================================================

BASE_DIR = Path(r"D:\Backend\serenova-backend")

EMBED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "ctu_uhb_ctg"
    / "embeddings"
)

RESULTS_DIR = (
    BASE_DIR
    / "data"
    / "results"
    / "ctu_uhb_ctg"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_embedding(split):
    path = EMBED_DIR / f"{split}_embeddings.npy"

    if not path.exists():
        raise FileNotFoundError(f"Missing embedding file: {path}")

    x = np.load(path)

    if x.ndim != 3:
        raise ValueError(
            f"{split}: expected [N,C,T], got {x.shape}"
        )

    return x


def summarize_split(name, x):
    n, c, t = x.shape

    channel_stats = []

    for ch in range(c):
        z = x[:, ch, :]

        channel_stats.append(
            {
                "channel": int(ch),
                "mean": float(np.mean(z)),
                "std": float(np.std(z)),
                "min": float(np.min(z)),
                "max": float(np.max(z)),
                "mean_abs": float(np.mean(np.abs(z))),
                "mean_temporal_std": float(
                    np.mean(np.std(z, axis=1))
                ),
                "mean_window_energy": float(
                    np.mean(np.mean(z ** 2, axis=1))
                ),
            }
        )

    # Per-window representation energy.
    window_energy = np.mean(x ** 2, axis=(1, 2))

    # Per-channel temporal variance.
    temporal_std = np.std(x, axis=2)

    # Detect channels with effectively no temporal variation.
    collapsed_channels = []

    for ch in range(c):
        if np.mean(temporal_std[:, ch]) < 1e-6:
            collapsed_channels.append(ch)

    result = {
        "split": name,
        "shape": list(x.shape),
        "finite": bool(np.isfinite(x).all()),
        "global": {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean_abs": float(np.mean(np.abs(x))),
            "window_energy_mean": float(np.mean(window_energy)),
            "window_energy_std": float(np.std(window_energy)),
            "window_energy_min": float(np.min(window_energy)),
            "window_energy_max": float(np.max(window_energy)),
        },
        "channel_stats": channel_stats,
        "collapsed_channels": collapsed_channels,
        "mean_temporal_std": float(np.mean(temporal_std)),
        "temporal_std_min": float(np.min(temporal_std)),
        "temporal_std_max": float(np.max(temporal_std)),
    }

    return result


def distribution_distance(a, b):
    return {
        "mean_abs_difference": float(
            abs(np.mean(a) - np.mean(b))
        ),
        "std_abs_difference": float(
            abs(np.std(a) - np.std(b))
        ),
        "mean_absolute_value_difference": float(
            abs(
                np.mean(np.abs(a))
                - np.mean(np.abs(b))
            )
        ),
    }


def pooled_statistics(x):
    """
    Statistics after global temporal pooling.

    This approximates the information available to a simple
    non-temporal baseline.
    """
    pooled = np.mean(x, axis=2)

    return {
        "shape": list(pooled.shape),
        "mean": float(np.mean(pooled)),
        "std": float(np.std(pooled)),
        "min": float(np.min(pooled)),
        "max": float(np.max(pooled)),
        "finite": bool(np.isfinite(pooled).all()),
    }


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

print("=" * 70)
print("CTU-UHB CTG EMBEDDING DIAGNOSTICS")
print("=" * 70)

train = load_embedding("train")
val = load_embedding("val")
test = load_embedding("test")

print(f"TRAIN: {train.shape}")
print(f"VAL:   {val.shape}")
print(f"TEST:  {test.shape}")


# ------------------------------------------------------------
# Basic diagnostics
# ------------------------------------------------------------

diagnostics = {
    "dataset": "ctu_uhb_ctg",
    "embedding_source": str(EMBED_DIR),
    "train": summarize_split("train", train),
    "val": summarize_split("val", val),
    "test": summarize_split("test", test),
}


# ------------------------------------------------------------
# Train/validation/test distribution comparison
# ------------------------------------------------------------

diagnostics["distribution_shift"] = {
    "train_vs_val": distribution_distance(train, val),
    "train_vs_test": distribution_distance(train, test),
}


# ------------------------------------------------------------
# Global temporal pooling diagnostics
# ------------------------------------------------------------

diagnostics["temporal_pooling"] = {
    "train": pooled_statistics(train),
    "val": pooled_statistics(val),
    "test": pooled_statistics(test),
}


# ------------------------------------------------------------
# Channel-wise train/val/test comparison
# ------------------------------------------------------------

channel_shift = []

channels = train.shape[1]

for ch in range(channels):
    train_ch = train[:, ch, :]
    val_ch = val[:, ch, :]
    test_ch = test[:, ch, :]

    channel_shift.append(
        {
            "channel": int(ch),
            "train_vs_val_mean_difference": float(
                abs(
                    np.mean(train_ch)
                    - np.mean(val_ch)
                )
            ),
            "train_vs_test_mean_difference": float(
                abs(
                    np.mean(train_ch)
                    - np.mean(test_ch)
                )
            ),
            "train_vs_val_std_difference": float(
                abs(
                    np.std(train_ch)
                    - np.std(val_ch)
                )
            ),
            "train_vs_test_std_difference": float(
                abs(
                    np.std(train_ch)
                    - np.std(test_ch)
                )
            ),
        }
    )

diagnostics["channel_distribution_shift"] = channel_shift


# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

for split_name in ["train", "val", "test"]:
    s = diagnostics[split_name]

    print()
    print(f"{split_name.upper()}")
    print("-" * 50)
    print(f"shape:              {s['shape']}")
    print(f"finite:             {s['finite']}")
    print(f"mean:               {s['global']['mean']:.6f}")
    print(f"std:                {s['global']['std']:.6f}")
    print(f"min:                {s['global']['min']:.6f}")
    print(f"max:                {s['global']['max']:.6f}")
    print(
        f"mean temporal std:  "
        f"{s['mean_temporal_std']:.6f}"
    )
    print(
        f"energy mean:        "
        f"{s['global']['window_energy_mean']:.6f}"
    )
    print(
        f"energy std:         "
        f"{s['global']['window_energy_std']:.6f}"
    )
    print(
        f"collapsed channels: "
        f"{s['collapsed_channels']}"
    )


print()
print("DISTRIBUTION SHIFT")
print("-" * 50)

print(
    "Train vs Val mean difference:",
    f"{diagnostics['distribution_shift']['train_vs_val']['mean_abs_difference']:.6f}"
)

print(
    "Train vs Test mean difference:",
    f"{diagnostics['distribution_shift']['train_vs_test']['mean_abs_difference']:.6f}"
)

print(
    "Train vs Val std difference:",
    f"{diagnostics['distribution_shift']['train_vs_val']['std_abs_difference']:.6f}"
)

print(
    "Train vs Test std difference:",
    f"{diagnostics['distribution_shift']['train_vs_test']['std_abs_difference']:.6f}"
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output_path = RESULTS_DIR / "embedding_analysis.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        diagnostics,
        f,
        indent=2
    )

print()
print("=" * 70)
print(f"Saved diagnostics: {output_path}")
print("=" * 70)