"""
SERENOVA — CTU-UHB CTG EMBEDDING EXTRACTION

Purpose
-------
Load the already-trained CTU-UHB CTG 1D CNN autoencoder,
freeze it, and extract latent temporal representations.

IMPORTANT
---------
This script does NOT train another CNN.

Existing model:
    data/results/ctu_uhb_ctg/model/
        ctu_uhb_ctg_1dcnn_autoencoder.pt

Input:
    data/processed/ctu_uhb_ctg/
        train/X.npy
        val/X.npy
        test/X.npy

Expected input:
    [N, 2, 1200]

Expected latent:
    [N, 64, 150]

Output:
    data/processed/ctu_uhb_ctg/embeddings/
        train_embeddings.npy
        val_embeddings.npy
        test_embeddings.npy
        metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "ctu_uhb_ctg"
)

MODEL_PATH = (
    BASE_DIR
    / "data"
    / "results"
    / "ctu_uhb_ctg"
    / "model"
    / "ctu_uhb_ctg_1dcnn_autoencoder.pt"
)

EMBEDDING_DIR = (
    PROCESSED_DIR
    / "embeddings"
)

EMBEDDING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 64

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL
# ============================================================

class CTUWaveformAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv1d(
                2,
                32,
                kernel_size=7,
                stride=2,
                padding=3,
            ),

            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),

            nn.Conv1d(
                32,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
            ),

            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),

            nn.Conv1d(
                64,
                128,
                kernel_size=7,
                stride=2,
                padding=3,
            ),

            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            nn.Conv1d(
                128,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose1d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose1d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose1d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),

            nn.Conv1d(
                32,
                2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.encoder(x)

    def decode(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:

        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        z = self.encode(x)

        reconstruction = self.decode(z)

        if reconstruction.shape[-1] != x.shape[-1]:

            reconstruction = nn.functional.interpolate(
                reconstruction,
                size=x.shape[-1],
                mode="linear",
                align_corners=False,
            )

        return reconstruction


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"CTU model not found:\n{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = CTUWaveformAutoencoder()

    # --------------------------------------------------------
    # Support both:
    #
    # checkpoint["model_state_dict"]
    #
    # and a direct state_dict.
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(
        DEVICE
    )

    # IMPORTANT:
    # We are extracting embeddings only.
    # No training occurs.
    model.eval()

    for parameter in model.parameters():

        parameter.requires_grad = False

    return model, checkpoint


# ============================================================
# LOAD DATA
# ============================================================

def load_split(
    split: str,
):

    path = (
        PROCESSED_DIR
        / split
        / "X.npy"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing processed data:\n{path}"
        )

    x = np.load(
        path,
        mmap_mode="r",
    )

    print()
    print(
        f"{split.upper()} input:"
    )

    print(
        f"  Shape: {x.shape}"
    )

    print(
        f"  Dtype: {x.dtype}"
    )

    if x.ndim != 3:

        raise ValueError(
            f"{split}: expected [N,C,T], "
            f"got {x.shape}"
        )

    if x.shape[1] != 2:

        raise ValueError(
            f"{split}: expected 2 channels, "
            f"got {x.shape[1]}"
        )

    if x.shape[2] != 1200:

        raise ValueError(
            f"{split}: expected 1200 samples, "
            f"got {x.shape[2]}"
        )

    if not np.isfinite(x).all():

        raise ValueError(
            f"{split}: NaN/Inf detected"
        )

    # Copy mmap data to writable float32 array.
    x = np.array(
        x,
        dtype=np.float32,
        copy=True,
    )

    return torch.from_numpy(x)


# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

@torch.no_grad()
def extract_embeddings(
    model,
    x: torch.Tensor,
):

    dataset = TensorDataset(
        x
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    embeddings = []

    for batch_index, (batch,) in enumerate(
        loader,
        start=1,
    ):

        batch = batch.to(
            DEVICE,
            non_blocking=True,
        )

        latent = model.encode(
            batch
        )

        # ----------------------------------------------------
        # Expected:
        # [B, 64, 150]
        # ----------------------------------------------------

        if latent.ndim != 3:

            raise RuntimeError(
                "Latent representation must be 3D. "
                f"Got {latent.shape}"
            )

        if latent.shape[1] != 64:

            raise RuntimeError(
                "Expected 64 latent channels. "
                f"Got {latent.shape}"
            )

        if latent.shape[2] != 150:

            raise RuntimeError(
                "Expected temporal length 150. "
                f"Got {latent.shape}"
            )

        embeddings.append(
            latent.cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        if (
            batch_index == 1
            or batch_index % 10 == 0
        ):

            print(
                f"  Processed batch "
                f"{batch_index}"
            )

    embeddings = np.concatenate(
        embeddings,
        axis=0,
    )

    if not np.isfinite(
        embeddings
    ).all():

        raise RuntimeError(
            "Non-finite values found in embeddings."
        )

    return embeddings


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    model_checkpoint,
    shapes,
):

    metadata = {

        "dataset":
            "ctu_uhb_ctg",

        "task":
            "cnn_latent_embedding_extraction",

        "source_model":
            "CTUWaveformAutoencoder",

        "checkpoint_path":
            str(MODEL_PATH),

        "device":
            str(DEVICE),

        "frozen_encoder":
            True,

        "training_performed":
            False,

        "input_shape":
            [
                2,
                1200,
            ],

        "latent_shape":
            [
                64,
                150,
            ],

        "sampling_frequency":
            "inherited_from_processed_dataset",

        "window_length_samples":
            1200,

        "channels": [
            "FHR",
            "UC",
        ],

        "split_shapes":
            shapes,

        "normalization":
            "inherited_from_processed_ctu_dataset",

        "embedding_directory":
            str(EMBEDDING_DIR),
    }

    if isinstance(
        model_checkpoint,
        dict
    ):

        if "best_epoch" in model_checkpoint:

            metadata[
                "source_model_best_epoch"
            ] = int(
                model_checkpoint[
                    "best_epoch"
                ]
            )

        if "best_val_loss" in model_checkpoint:

            metadata[
                "source_model_best_val_loss"
            ] = float(
                model_checkpoint[
                    "best_val_loss"
                ]
            )

        if "parameter_count" in model_checkpoint:

            metadata[
                "source_model_parameter_count"
            ] = int(
                model_checkpoint[
                    "parameter_count"
                ]
            )

    metadata_path = (
        EMBEDDING_DIR
        / "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    return metadata_path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("SERENOVA — CTU-UHB CTG EMBEDDING EXTRACTION")
    print("=" * 72)

    print()
    print(
        f"Device: {DEVICE}"
    )

    print()
    print(
        "Model:"
    )

    print(
        MODEL_PATH
    )

    print()
    print(
        "Output:"
    )

    print(
        EMBEDDING_DIR
    )

    # --------------------------------------------------------
    # LOAD FROZEN MODEL
    # --------------------------------------------------------

    model, checkpoint = load_model()

    print()
    print(
        "Frozen encoder loaded successfully."
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    splits = {}

    for split in [
        "train",
        "val",
        "test",
    ]:

        splits[split] = load_split(
            split
        )

    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------

    saved_shapes = {}

    for split in [
        "train",
        "val",
        "test",
    ]:

        print()
        print("=" * 72)
        print(
            f"EXTRACTING {split.upper()} EMBEDDINGS"
        )
        print("=" * 72)

        embeddings = extract_embeddings(
            model,
            splits[split],
        )

        output_path = (
            EMBEDDING_DIR
            / f"{split}_embeddings.npy"
        )

        np.save(
            output_path,
            embeddings,
        )

        saved_shapes[
            split
        ] = list(
            embeddings.shape
        )

        print()
        print(
            f"{split.upper()} embedding shape:"
        )

        print(
            embeddings.shape
        )

        print(
            f"Finite: {np.isfinite(embeddings).all()}"
        )

        print(
            f"Saved: {output_path}"
        )

        # Free memory.
        del embeddings

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    metadata_path = save_metadata(
        checkpoint,
        saved_shapes,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("CTU-UHB CTG EMBEDDING EXTRACTION COMPLETE")
    print("=" * 72)

    print()

    for split, shape in saved_shapes.items():

        print(
            f"{split}: {shape}"
        )

    print()
    print(
        "Metadata:"
    )

    print(
        metadata_path
    )

    print()
    print(
        "No model training was performed."
    )


if __name__ == "__main__":
    main()