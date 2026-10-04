"""
SERENOVA — CTU-UHB CTG 1D CNN AUTOENCODER

Task:
    Unsupervised waveform reconstruction / representation learning.

Input:
    [N, 2, 1200]

Latent:
    [N, 64, 150]

Output:
    [N, 2, 1200]

Existing processed data:
    data/processed/ctu_uhb_ctg/train/X.npy
    data/processed/ctu_uhb_ctg/val/X.npy
    data/processed/ctu_uhb_ctg/test/X.npy

Saved model:
    data/results/ctu_uhb_ctg/model/
        ctu_uhb_ctg_1dcnn_autoencoder.pt

Saved metrics:
    data/results/ctu_uhb_ctg/
        baseline_metrics.json
"""

from __future__ import annotations

import json
import random
import time
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

RESULTS_DIR = (
    BASE_DIR
    / "data"
    / "results"
    / "ctu_uhb_ctg"
)

MODEL_DIR = RESULTS_DIR / "model"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRAIN_PATH = PROCESSED_DIR / "train" / "X.npy"
VAL_PATH = PROCESSED_DIR / "val" / "X.npy"
TEST_PATH = PROCESSED_DIR / "test" / "X.npy"


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 64
EPOCHS = 25

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 6

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATA LOADING
# ============================================================

def load_data(path: Path) -> torch.Tensor:

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )

    x = np.load(
        path,
        mmap_mode="r",
    )

    print(
        f"Loading: {path}"
    )

    print(
        f"Shape: {x.shape}"
    )

    if x.ndim != 3:
        raise ValueError(
            f"Expected [N,2,1200], got {x.shape}"
        )

    if x.shape[1] != 2:
        raise ValueError(
            f"Expected 2 channels, got {x.shape[1]}"
        )

    if x.shape[2] != 1200:
        raise ValueError(
            f"Expected 1200 samples, got {x.shape[2]}"
        )

    if not np.isfinite(x).all():
        raise ValueError(
            f"NaN/Inf detected in {path}"
        )

    # Copy because mmap arrays may be non-writable.
    x = np.array(
        x,
        dtype=np.float32,
        copy=True,
    )

    return torch.from_numpy(x)


# ============================================================
# MODEL
# ============================================================

class CTUWaveformAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Encoder
        #
        # [B,2,1200]
        #       ↓
        # [B,32,600]
        #       ↓
        # [B,64,300]
        #       ↓
        # [B,128,150]
        #       ↓
        # [B,64,150]
        # ----------------------------------------------------

        self.encoder = nn.Sequential(

            nn.Conv1d(
                in_channels=2,
                out_channels=32,
                kernel_size=7,
                stride=2,
                padding=3,
            ),

            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
            ),

            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=7,
                stride=2,
                padding=3,
            ),

            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            nn.Conv1d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
        )

        # ----------------------------------------------------
        # Decoder
        #
        # [B,64,150]
        #       ↓
        # [B,128,300]
        #       ↓
        # [B,64,600]
        #       ↓
        # [B,32,1200]
        #       ↓
        # [B,2,1200]
        # ----------------------------------------------------

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

    def encode(self, x):

        return self.encoder(x)

    def decode(self, z):

        return self.decoder(z)

    def forward(self, x):

        z = self.encode(x)

        reconstruction = self.decode(z)

        # Safety correction.
        if reconstruction.shape[-1] != x.shape[-1]:

            reconstruction = nn.functional.interpolate(
                reconstruction,
                size=x.shape[-1],
                mode="linear",
                align_corners=False,
            )

        return reconstruction


# ============================================================
# TRAINING
# ============================================================

def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
):

    model.train()

    total_loss = 0.0
    batches = 0

    for (x,) in loader:

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        reconstruction = model(x)

        loss = criterion(
            reconstruction,
            x,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()
        batches += 1

    return total_loss / batches


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validation_epoch(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0
    batches = 0

    for (x,) in loader:

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        reconstruction = model(x)

        loss = criterion(
            reconstruction,
            x,
        )

        total_loss += loss.item()
        batches += 1

    return total_loss / batches


# ============================================================
# TEST METRICS
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    loader,
):

    model.eval()

    predictions = []
    targets = []

    for (x,) in loader:

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        prediction = model(x)

        predictions.append(
            prediction.cpu()
        )

        targets.append(
            x.cpu()
        )

    prediction = torch.cat(
        predictions,
        dim=0,
    )

    target = torch.cat(
        targets,
        dim=0,
    )

    error = prediction - target

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    mse = torch.mean(
        error ** 2
    ).item()

    rmse = float(
        np.sqrt(mse)
    )

    mae = torch.mean(
        torch.abs(error)
    ).item()

    signal_power = torch.mean(
        target ** 2
    ).item()

    noise_power = torch.mean(
        error ** 2
    ).item()

    snr_db = float(
        10.0
        * np.log10(
            signal_power
            /
            max(noise_power, 1e-12)
        )
    )

    # --------------------------------------------------------
    # Per-channel metrics
    # --------------------------------------------------------

    channel_metrics = {}

    for channel in range(
        target.shape[1]
    ):

        y_true = (
            target[:, channel, :]
            .reshape(-1)
            .numpy()
        )

        y_pred = (
            prediction[:, channel, :]
            .reshape(-1)
            .numpy()
        )

        channel_error = (
            y_pred - y_true
        )

        channel_mse = float(
            np.mean(
                channel_error ** 2
            )
        )

        channel_rmse = float(
            np.sqrt(channel_mse)
        )

        channel_mae = float(
            np.mean(
                np.abs(channel_error)
            )
        )

        if (
            np.std(y_true) > 1e-12
            and
            np.std(y_pred) > 1e-12
        ):

            pearson_r = float(
                np.corrcoef(
                    y_true,
                    y_pred,
                )[0, 1]
            )

        else:

            pearson_r = None

        channel_metrics[
            f"channel_{channel}"
        ] = {
            "mse": channel_mse,
            "rmse": channel_rmse,
            "mae": channel_mae,
            "pearson_r": pearson_r,
        }

    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mae),
        "snr_db": snr_db,
        "channel_metrics": channel_metrics,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print()
    print("=" * 72)
    print("SERENOVA — CTU-UHB CTG 1D CNN AUTOENCODER")
    print("=" * 72)

    print()
    print(f"Device: {DEVICE}")
    print(f"Processed directory:")
    print(PROCESSED_DIR)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    X_train = load_data(
        TRAIN_PATH
    )

    X_val = load_data(
        VAL_PATH
    )

    X_test = load_data(
        TEST_PATH
    )

    print()
    print("=" * 72)
    print("DATASET SUMMARY")
    print("=" * 72)

    print(
        f"Train: {tuple(X_train.shape)}"
    )

    print(
        f"Val:   {tuple(X_val.shape)}"
    )

    print(
        f"Test:  {tuple(X_test.shape)}"
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        X_train
    )

    val_dataset = TensorDataset(
        X_val
    )

    test_dataset = TensorDataset(
        X_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = CTUWaveformAutoencoder().to(
        DEVICE
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Trainable parameters: {parameter_count:,}"
    )

    # --------------------------------------------------------
    # ARCHITECTURE CHECK
    # --------------------------------------------------------

    with torch.no_grad():

        sample = X_train[:2].to(
            DEVICE
        )

        latent = model.encode(
            sample
        )

        reconstruction = model(
            sample
        )

    print()
    print("Architecture check:")

    print(
        f"Input:          {tuple(sample.shape)}"
    )

    print(
        f"Latent:         {tuple(latent.shape)}"
    )

    print(
        f"Reconstruction: {tuple(reconstruction.shape)}"
    )

    if reconstruction.shape != sample.shape:

        raise RuntimeError(
            "Reconstruction shape does not match input."
        )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_val_loss = float("inf")

    best_epoch = 0

    best_state = None

    patience_counter = 0

    history = []

    start_time = time.time()

    print()
    print("=" * 72)
    print("TRAINING")
    print("=" * 72)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss = validation_epoch(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(
            val_loss
        )

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f} | "
            f"lr={current_lr:.2e}"
        )

        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch

            patience_counter = 0

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

        else:

            patience_counter += 1

            if (
                patience_counter
                >= EARLY_STOPPING_PATIENCE
            ):

                print()
                print(
                    "Early stopping."
                )

                break

    training_seconds = (
        time.time()
        - start_time
    )

    if best_state is None:

        raise RuntimeError(
            "No best checkpoint was captured."
        )

    # --------------------------------------------------------
    # RESTORE BEST MODEL
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    # --------------------------------------------------------
    # LOCKED TEST
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("LOCKED TEST EVALUATION")
    print("=" * 72)

    test_metrics = evaluate_model(
        model,
        test_loader,
    )

    print()
    print(
        f"Overall MSE  : {test_metrics['mse']:.6f}"
    )

    print(
        f"Overall RMSE : {test_metrics['rmse']:.6f}"
    )

    print(
        f"Overall MAE  : {test_metrics['mae']:.6f}"
    )

    print(
        f"SNR          : {test_metrics['snr_db']:.3f} dB"
    )

    for channel, metrics in (
        test_metrics[
            "channel_metrics"
        ].items()
    ):

        print()
        print(channel)

        print(
            f"  MSE       : {metrics['mse']:.6f}"
        )

        print(
            f"  RMSE      : {metrics['rmse']:.6f}"
        )

        print(
            f"  MAE       : {metrics['mae']:.6f}"
        )

        print(
            f"  Pearson R : {metrics['pearson_r']}"
        )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "ctu_uhb_ctg_1dcnn_autoencoder.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "architecture":
                "CTUWaveformAutoencoder",

            "input_shape":
                [2, 1200],

            "latent_shape":
                [64, 150],

            "seed":
                SEED,

            "best_epoch":
                best_epoch,

            "best_val_loss":
                float(best_val_loss),

            "parameter_count":
                parameter_count,

            "processed_data":
                str(PROCESSED_DIR),
        },
        model_path,
    )

    # --------------------------------------------------------
    # SAVE METADATA / METRICS
    # --------------------------------------------------------

    metrics_path = (
        RESULTS_DIR
        / "baseline_metrics.json"
    )

    report = {

        "dataset":
            "ctu_uhb_ctg",

        "task":
            "unsupervised_waveform_reconstruction",

        "status":
            "experimental_waveform_reconstruction_baseline",

        "seed":
            SEED,

        "device":
            str(DEVICE),

        "input_shape":
            [2, 1200],

        "latent_shape":
            [64, 150],

        "parameter_count":
            parameter_count,

        "train_samples":
            int(X_train.shape[0]),

        "val_samples":
            int(X_val.shape[0]),

        "test_samples":
            int(X_test.shape[0]),

        "best_epoch":
            best_epoch,

        "best_val_loss":
            float(best_val_loss),

        "training_seconds":
            float(training_seconds),

        "history":
            history,

        "locked_test_metrics":
            test_metrics,

        "model_path":
            str(model_path),

        "processed_data":
            str(PROCESSED_DIR),

        "note":
            (
                "This is an unsupervised CTU-UHB CTG "
                "waveform representation baseline. "
                "No clinical target is assigned at "
                "this stage."
            ),
    }

    with open(
        metrics_path,
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
    print("CTU-UHB CTG TRAINING COMPLETE")
    print("=" * 72)

    print()
    print("Best epoch:")
    print(best_epoch)

    print()
    print("Model saved:")
    print(model_path)

    print()
    print("Metrics saved:")
    print(metrics_path)


if __name__ == "__main__":
    main()