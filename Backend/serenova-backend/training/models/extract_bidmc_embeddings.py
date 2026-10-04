from pathlib import Path
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from training.datasets.pytorch_datasets import SerenovaNumpyDataset


PROCESSED_DIR = BASE_DIR / "data" / "processed" / "bidmc"
RESULTS_DIR = BASE_DIR / "data" / "results" / "bidmc"
MODEL_PATH = (
    RESULTS_DIR
    / "model"
    / "bidmc_1dcnn_autoencoder.pt"
)

EMBEDDING_DIR = (
    PROCESSED_DIR
    / "embeddings"
)

EMBEDDING_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL
# ============================================================

class Waveform1DCNNAutoencoder(nn.Module):
    """
    Same architecture used by baseline_bidmc.py.

    Input:
        [B, 2, 1250]

    Encoder output:
        [B, 64, 157]
    """

    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 64
    ):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv1d(
                in_channels,
                32,
                kernel_size=7,
                stride=2,
                padding=3
            ),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Conv1d(
                32,
                64,
                kernel_size=7,
                stride=2,
                padding=3
            ),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Conv1d(
                64,
                128,
                kernel_size=7,
                stride=2,
                padding=3
            ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Conv1d(
                128,
                latent_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(
                0.2,
                inplace=True
            )
        )

        # Decoder is included so that the checkpoint
        # can be loaded exactly as saved by baseline_bidmc.py.
        self.decoder = nn.Sequential(

            nn.ConvTranspose1d(
                latent_dim,
                128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.ConvTranspose1d(
                128,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=0
            ),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.ConvTranspose1d(
                64,
                32,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=0
            ),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.ConvTranspose1d(
                32,
                in_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=1
            )
        )

    def encode(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.encoder(x)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        latent = self.encoder(x)

        reconstruction = self.decoder(
            latent
        )

        return reconstruction


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(
    dataset,
    split_name: str
):
    expected_shape = (2, 1250)

    if dataset.X.ndim != 3:
        raise ValueError(
            f"{split_name}: expected 3D array, "
            f"got {dataset.X.shape}"
        )

    if dataset.X.shape[1:] != expected_shape:
        raise ValueError(
            f"{split_name}: expected "
            f"[N, 2, 1250], "
            f"got {dataset.X.shape}"
        )

    if not np.isfinite(dataset.X).all():
        raise ValueError(
            f"{split_name}: "
            f"non-finite values detected"
        )

    print(
        f"{split_name:<5} "
        f"X={dataset.X.shape} "
        f"dtype={dataset.X.dtype} "
        f"finite=PASS"
    )


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================

def extract_embeddings(
    model,
    dataset,
    split_name,
    device,
    batch_size=32
):
    """
    Extract CNN latent representation.

    Input:
        [N, 2, 1250]

    Output:
        [N, 64, 157]
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    embeddings = []

    model.eval()

    with torch.no_grad():

        for batch_idx, batch in enumerate(loader):

            x = batch.to(
                device,
                non_blocking=True
            )

            latent = model.encode(x)

            latent_cpu = (
                latent
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False
                )
            )

            embeddings.append(
                latent_cpu
            )

            if (
                batch_idx == 0
                or (batch_idx + 1) % 10 == 0
            ):
                print(
                    f"{split_name}: "
                    f"processed batch "
                    f"{batch_idx + 1}"
                )

    result = np.concatenate(
        embeddings,
        axis=0
    )

    expected_shape = (
        len(dataset),
        64,
        157
    )

    if result.shape != expected_shape:
        raise RuntimeError(
            f"{split_name}: unexpected embedding "
            f"shape {result.shape}; "
            f"expected {expected_shape}"
        )

    if not np.isfinite(result).all():
        raise RuntimeError(
            f"{split_name}: "
            f"embedding contains non-finite values"
        )

    print(
        f"{split_name}: "
        f"embedding shape = {result.shape}"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("SERENOVA — BIDMC CNN EMBEDDING EXTRACTION")
    print("=" * 72)

    print(
        f"Project root : {BASE_DIR}"
    )

    print(
        f"Processed    : {PROCESSED_DIR}"
    )

    print(
        f"Model        : {MODEL_PATH}"
    )

    print(
        f"Output       : {EMBEDDING_DIR}"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device       : {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "BIDMC autoencoder checkpoint "
            f"not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = SerenovaNumpyDataset(
        "bidmc",
        split="train"
    )

    val_dataset = SerenovaNumpyDataset(
        "bidmc",
        split="val"
    )

    test_dataset = SerenovaNumpyDataset(
        "bidmc",
        split="test"
    )

    print("\nDataset validation:")

    validate_dataset(
        train_dataset,
        "TRAIN"
    )

    validate_dataset(
        val_dataset,
        "VAL"
    )

    validate_dataset(
        test_dataset,
        "TEST"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading trained autoencoder...")

    model = Waveform1DCNNAutoencoder(
        in_channels=2,
        latent_dim=64
    ).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    print(
        "Checkpoint load: PASS"
    )

    # --------------------------------------------------------
    # Extract embeddings
    # --------------------------------------------------------

    print("\nExtracting TRAIN embeddings...")

    train_embeddings = extract_embeddings(
        model,
        train_dataset,
        "TRAIN",
        device
    )

    print("\nExtracting VAL embeddings...")

    val_embeddings = extract_embeddings(
        model,
        val_dataset,
        "VAL",
        device
    )

    print("\nExtracting TEST embeddings...")

    test_embeddings = extract_embeddings(
        model,
        test_dataset,
        "TEST",
        device
    )

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    train_path = (
        EMBEDDING_DIR
        / "train_embeddings.npy"
    )

    val_path = (
        EMBEDDING_DIR
        / "val_embeddings.npy"
    )

    test_path = (
        EMBEDDING_DIR
        / "test_embeddings.npy"
    )

    np.save(
        train_path,
        train_embeddings
    )

    np.save(
        val_path,
        val_embeddings
    )

    np.save(
        test_path,
        test_embeddings
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {

        "dataset": "bidmc",

        "source_model":
            "Waveform1DCNNAutoencoder",

        "checkpoint_path":
            str(MODEL_PATH),

        "task":
            "cnn_latent_embedding_extraction",

        "input_shape":
            [2, 1250],

        "sampling_rate_hz":
            125,

        "window_duration_seconds":
            10,

        "channel_mapping": {
            "0": "PLETH_PPG",
            "1": "RESP"
        },

        "latent_channels":
            64,

        "latent_temporal_length":
            157,

        "latent_shape":
            [64, 157],

        "train_shape":
            list(
                train_embeddings.shape
            ),

        "val_shape":
            list(
                val_embeddings.shape
            ),

        "test_shape":
            list(
                test_embeddings.shape
            ),

        "dtype":
            "float32",

        "train_records":
            37,

        "val_records":
            8,

        "test_records":
            8,

        "record_split_policy":
            "locked_record_level_split",

        "normalization":
            "inherited_from_processed_BIDMC_inputs",

        "files": {
            "train":
                str(train_path),
            "val":
                str(val_path),
            "test":
                str(test_path)
        },

        "status":
            "embedding_extraction_complete"
    }

    metadata_path = (
        EMBEDDING_DIR
        / "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("EMBEDDING EXTRACTION VERIFICATION")
    print("=" * 72)

    print(
        f"TRAIN: {train_embeddings.shape}"
    )

    print(
        f"VAL  : {val_embeddings.shape}"
    )

    print(
        f"TEST : {test_embeddings.shape}"
    )

    print(
        f"TRAIN finite: "
        f"{np.isfinite(train_embeddings).all()}"
    )

    print(
        f"VAL finite: "
        f"{np.isfinite(val_embeddings).all()}"
    )

    print(
        f"TEST finite: "
        f"{np.isfinite(test_embeddings).all()}"
    )

    print(
        f"\nSaved to:\n"
        f"{EMBEDDING_DIR}"
    )

    print(
        "\nFiles:"
    )

    print(
        f"  {train_path.name}"
    )

    print(
        f"  {val_path.name}"
    )

    print(
        f"  {test_path.name}"
    )

    print(
        f"  {metadata_path.name}"
    )

    print("=" * 72)
    print("STATUS: BIDMC EMBEDDING EXTRACTION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()