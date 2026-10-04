from pathlib import Path
import json
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]

EMBEDDING_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "bidmc"
    / "embeddings"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "results"
    / "bidmc"
    / "embedding_analysis.json"
)


def analyze_split(name: str):
    path = EMBEDDING_DIR / f"{name}_embeddings.npy"

    x = np.load(
        path,
        mmap_mode="r"
    )

    if x.ndim != 3:
        raise ValueError(
            f"{name}: expected [N,64,157], got {x.shape}"
        )

    if x.shape[1:] != (64, 157):
        raise ValueError(
            f"{name}: unexpected shape {x.shape}"
        )

    finite = bool(np.isfinite(x).all())

    if not finite:
        raise ValueError(
            f"{name}: non-finite values detected"
        )

    # Global statistics
    global_mean = float(np.mean(x))
    global_std = float(np.std(x))
    global_min = float(np.min(x))
    global_max = float(np.max(x))

    # Per-channel statistics across samples and time
    channel_mean = np.mean(
        x,
        axis=(0, 2)
    )

    channel_std = np.std(
        x,
        axis=(0, 2)
    )

    channel_abs_mean = np.mean(
        np.abs(x),
        axis=(0, 2)
    )

    # Channel collapse:
    # very small standard deviation means a latent channel
    # may contain almost no variation.
    collapsed_threshold = 1e-4

    collapsed_channels = int(
        np.sum(
            channel_std < collapsed_threshold
        )
    )

    # Temporal variation
    temporal_std = np.std(
        x,
        axis=2
    )

    mean_temporal_std = float(
        np.mean(temporal_std)
    )

    # Per-window latent energy
    window_energy = np.mean(
        x.astype(np.float64) ** 2,
        axis=(1, 2)
    )

    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite": finite,

        "global": {
            "mean": global_mean,
            "std": global_std,
            "min": global_min,
            "max": global_max
        },

        "latent": {
            "channels": 64,
            "temporal_length": 157,

            "mean_channel_std": float(
                np.mean(channel_std)
            ),

            "min_channel_std": float(
                np.min(channel_std)
            ),

            "max_channel_std": float(
                np.max(channel_std)
            ),

            "collapsed_channels": collapsed_channels,

            "mean_temporal_std": mean_temporal_std
        },

        "window_energy": {
            "mean": float(
                np.mean(window_energy)
            ),
            "std": float(
                np.std(window_energy)
            ),
            "min": float(
                np.min(window_energy)
            ),
            "max": float(
                np.max(window_energy)
            )
        }
    }


def main():

    print("=" * 72)
    print("SERENOVA — BIDMC EMBEDDING ANALYSIS")
    print("=" * 72)

    print(
        f"Embedding directory:\n{EMBEDDING_DIR}"
    )

    results = {}

    for split in [
        "train",
        "val",
        "test"
    ]:

        print(
            f"\nAnalyzing {split.upper()}..."
        )

        results[split] = analyze_split(
            split
        )

        r = results[split]

        print(
            f"Shape: {r['shape']}"
        )

        print(
            f"Mean: {r['global']['mean']:.6f}"
        )

        print(
            f"Std : {r['global']['std']:.6f}"
        )

        print(
            f"Min : {r['global']['min']:.6f}"
        )

        print(
            f"Max : {r['global']['max']:.6f}"
        )

        print(
            "Collapsed latent channels: "
            f"{r['latent']['collapsed_channels']}"
        )

        print(
            "Mean temporal std: "
            f"{r['latent']['mean_temporal_std']:.6f}"
        )

    # --------------------------------------------------------
    # Train / validation / test distribution comparison
    # --------------------------------------------------------

    train_mean = results["train"]["global"]["mean"]
    val_mean = results["val"]["global"]["mean"]
    test_mean = results["test"]["global"]["mean"]

    train_std = results["train"]["global"]["std"]
    val_std = results["val"]["global"]["std"]
    test_std = results["test"]["global"]["std"]

    results["distribution_comparison"] = {
        "mean_difference_train_val": abs(
            train_mean - val_mean
        ),
        "mean_difference_train_test": abs(
            train_mean - test_mean
        ),
        "std_difference_train_val": abs(
            train_std - val_std
        ),
        "std_difference_train_test": abs(
            train_std - test_std
        )
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print("\n")
    print("=" * 72)
    print("EMBEDDING ANALYSIS COMPLETE")
    print("=" * 72)

    print(
        f"Saved:\n{OUTPUT_PATH}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()