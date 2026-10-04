from pathlib import Path
import sys
import json
import time

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from training.datasets.pytorch_datasets import SerenovaNumpyDataset


PROCESSED_DIR = BASE_DIR / "data" / "processed" / "bidmc"
RESULTS_DIR = BASE_DIR / "data" / "results" / "bidmc"
MODEL_DIR = RESULTS_DIR / "model"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# MODEL
# ============================================================

class Waveform1DCNNAutoencoder(nn.Module):
    """
    1D-CNN Autoencoder for dual-channel physiological waveforms.

    Input:
        [batch, 2, 1250]

    Encoder output:
        [batch, 64, 157]

    Decoder output:
        [batch, 2, 1250]

    Channel 0:
        PLETH_PPG

    Channel 1:
        RESP
    """

    def __init__(self, in_channels: int = 2, latent_dim: int = 64):
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder = nn.Sequential(

            # [B, 2, 1250]
            # -> [B, 32, 625]
            nn.Conv1d(
                in_channels,
                32,
                kernel_size=7,
                stride=2,
                padding=3
            ),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # [B, 32, 625]
            # -> [B, 64, 313]
            nn.Conv1d(
                32,
                64,
                kernel_size=7,
                stride=2,
                padding=3
            ),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # [B, 64, 313]
            # -> [B, 128, 157]
            nn.Conv1d(
                64,
                128,
                kernel_size=7,
                stride=2,
                padding=3
            ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # [B, 128, 157]
            # -> [B, 64, 157]
            nn.Conv1d(
                128,
                latent_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder = nn.Sequential(

            # [B, 64, 157]
            # -> [B, 128, 157]
            nn.ConvTranspose1d(
                latent_dim,
                128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # [B, 128, 157]
            # -> [B, 64, 313]
            nn.ConvTranspose1d(
                128,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=0
            ),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # [B, 64, 313]
            # -> [B, 32, 625]
            nn.ConvTranspose1d(
                64,
                32,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=0
            ),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # [B, 32, 625]
            # -> [B, 2, 1250]
            nn.ConvTranspose1d(
                32,
                in_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=1
            )
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the latent waveform representation.

        Input:
            [B, 2, 1250]

        Output:
            [B, 64, 157]
        """
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent representation back to waveform space.
        """
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Calculate reconstruction metrics.

    Expected shape:
        [N, 2, 1250]
    """

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, "
            f"y_pred={y_pred.shape}"
        )

    if y_true.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays [N, 2, 1250], "
            f"received {y_true.ndim}D"
        )

    if y_true.shape[1] != 2:
        raise ValueError(
            f"Expected 2 channels, received {y_true.shape[1]}"
        )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    error = y_true - y_pred

    mse_total = float(np.mean(error ** 2))
    rmse_total = float(np.sqrt(mse_total))
    mae_total = float(np.mean(np.abs(error)))

    # --------------------------------------------------------
    # PPG
    # --------------------------------------------------------

    ppg_true = y_true[:, 0, :]
    ppg_pred = y_pred[:, 0, :]

    ppg_error = ppg_true - ppg_pred

    ppg_mse = float(np.mean(ppg_error ** 2))
    ppg_mae = float(np.mean(np.abs(ppg_error)))

    ppg_corrs = []

    for i in range(len(y_true)):

        std_product = (
            np.std(ppg_true[i]) *
            np.std(ppg_pred[i])
        )

        if std_product > 1e-6:

            r = np.corrcoef(
                ppg_true[i],
                ppg_pred[i]
            )[0, 1]

            if np.isfinite(r):
                ppg_corrs.append(r)

    ppg_mean_r = (
        float(np.mean(ppg_corrs))
        if ppg_corrs
        else 0.0
    )

    # --------------------------------------------------------
    # Respiration
    # --------------------------------------------------------

    resp_true = y_true[:, 1, :]
    resp_pred = y_pred[:, 1, :]

    resp_error = resp_true - resp_pred

    resp_mse = float(np.mean(resp_error ** 2))
    resp_mae = float(np.mean(np.abs(resp_error)))

    resp_corrs = []

    for i in range(len(y_true)):

        std_product = (
            np.std(resp_true[i]) *
            np.std(resp_pred[i])
        )

        if std_product > 1e-6:

            r = np.corrcoef(
                resp_true[i],
                resp_pred[i]
            )[0, 1]

            if np.isfinite(r):
                resp_corrs.append(r)

    resp_mean_r = (
        float(np.mean(resp_corrs))
        if resp_corrs
        else 0.0
    )

    # --------------------------------------------------------
    # SNR
    # --------------------------------------------------------

    signal_power = float(np.mean(y_true ** 2))
    noise_power = float(np.mean(error ** 2)) + 1e-12

    snr_db = float(
        10.0 *
        np.log10(signal_power / noise_power)
    )

    return {
        "overall": {
            "mse": round(mse_total, 6),
            "rmse": round(rmse_total, 6),
            "mae": round(mae_total, 6),
            "snr_db": round(snr_db, 2)
        },

        "ppg_channel": {
            "channel_name": "PLETH_PPG",
            "mse": round(ppg_mse, 6),
            "mae": round(ppg_mae, 6),
            "pearson_r": round(ppg_mean_r, 4)
        },

        "respiration_channel": {
            "channel_name": "RESP",
            "mse": round(resp_mse, 6),
            "mae": round(resp_mae, 6),
            "pearson_r": round(resp_mean_r, 4)
        }
    }


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(dataset, split_name: str):

    expected_shape = (2, 1250)

    if dataset.X.ndim != 3:
        raise ValueError(
            f"{split_name}: expected 3D X array, "
            f"got shape {dataset.X.shape}"
        )

    if dataset.X.shape[1:] != expected_shape:
        raise ValueError(
            f"{split_name}: expected [N, 2, 1250], "
            f"got {dataset.X.shape}"
        )

    if not np.isfinite(dataset.X).all():
        raise ValueError(
            f"{split_name}: non-finite values detected"
        )

    print(
        f"{split_name:<6} "
        f"samples={len(dataset):4d} "
        f"shape={dataset.X.shape} "
        f"dtype={dataset.X.dtype} "
        f"finite=PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("SERENOVA — BIDMC 1D-CNN WAVEFORM BASELINE")
    print("=" * 72)

    print(f"Project root : {BASE_DIR}")
    print(f"Processed    : {PROCESSED_DIR}")
    print(f"Results      : {RESULTS_DIR}")
    print(f"Model dir    : {MODEL_DIR}")
    print(f"Random seed  : {SEED}")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device       : {device}")

    if torch.cuda.is_available():
        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Datasets
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

    print("\nDataset verification:")

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
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = Waveform1DCNNAutoencoder(
        in_channels=2,
        latent_dim=64
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nModel:")
    print("Architecture: Waveform1DCNNAutoencoder")
    print("Input       : [B, 2, 1250]")
    print("Latent      : [B, 64, 157]")
    print(f"Parameters  : {parameter_count:,}")
    print(
        f"Trainable   : "
        f"{trainable_parameter_count:,}"
    )

    # --------------------------------------------------------
    # Loss / optimizer
    # --------------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    epochs = 15

    best_val_loss = float("inf")

    checkpoint_path = (
        MODEL_DIR /
        "bidmc_1dcnn_autoencoder.pt"
    )

    print("\nTraining:")
    print("-" * 72)

    start_time = time.time()

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, epochs + 1):

        model.train()

        train_loss = 0.0

        for batch in train_loader:

            x = batch.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(x)

            loss = criterion(
                output,
                x
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item() *
                len(x)
            )

        train_loss /= len(train_dataset)

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for batch in val_loader:

                x = batch.to(
                    device,
                    non_blocking=True
                )

                output = model(x)

                loss = criterion(
                    output,
                    x
                )

                val_loss += (
                    loss.item() *
                    len(x)
                )

        val_loss /= len(val_dataset)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        is_best = val_loss < best_val_loss

        if is_best:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
            + (" | BEST" if is_best else "")
        )

    training_duration_s = round(
        time.time() - start_time,
        2
    )

    print("-" * 72)

    print(
        f"Training completed in "
        f"{training_duration_s}s"
    )

    print(
        f"Best validation MSE: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Expected checkpoint was not created: "
            f"{checkpoint_path}"
        )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    model.eval()

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print("\nEvaluating locked test set...")

    test_preds = []
    test_trues = []

    with torch.no_grad():

        for batch in test_loader:

            x = batch.to(
                device,
                non_blocking=True
            )

            output = model(x)

            test_preds.append(
                output.cpu().numpy()
            )

            test_trues.append(
                batch.numpy()
            )

    test_preds = np.concatenate(
        test_preds,
        axis=0
    )

    test_trues = np.concatenate(
        test_trues,
        axis=0
    )

    if test_preds.shape != test_trues.shape:
        raise RuntimeError(
            f"Prediction shape {test_preds.shape} "
            f"does not match target shape "
            f"{test_trues.shape}"
        )

    test_metrics = calculate_metrics(
        test_trues,
        test_preds
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    baseline_results = {

        "model_architecture":
            "Waveform1DCNNAutoencoder",

        "task":
            "unsupervised_waveform_representation_and_reconstruction",

        "dataset":
            "bidmc",

        "seed":
            SEED,

        "input_channels":
            2,

        "channel_mapping": {
            "0": "PLETH_PPG",
            "1": "RESP"
        },

        "input_window_samples":
            1250,

        "sampling_rate_hz":
            125,

        "window_duration_seconds":
            10,

        "latent_dim":
            64,

        "latent_shape":
            [64, 157],

        "parameters_count":
            parameter_count,

        "trainable_parameters_count":
            trainable_parameter_count,

        "training_epochs":
            epochs,

        "training_time_seconds":
            training_duration_s,

        "best_val_mse":
            round(
                best_val_loss,
                6
            ),

        "train_shape":
            list(train_dataset.X.shape),

        "val_shape":
            list(val_dataset.X.shape),

        "test_shape":
            list(test_dataset.X.shape),

        "test_metrics":
            test_metrics,

        "checkpoint_path":
            str(checkpoint_path),

        "status":
            "experimental_waveform_reconstruction_baseline"
    }

    metrics_path = (
        RESULTS_DIR /
        "baseline_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            baseline_results,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("BASELINE TEST SET RESULTS")
    print("=" * 72)

    print(
        f"Overall MSE       : "
        f"{test_metrics['overall']['mse']}"
    )

    print(
        f"Overall RMSE      : "
        f"{test_metrics['overall']['rmse']}"
    )

    print(
        f"Overall MAE       : "
        f"{test_metrics['overall']['mae']}"
    )

    print(
        f"Overall SNR       : "
        f"{test_metrics['overall']['snr_db']} dB"
    )

    print(
        f"PPG Pearson R     : "
        f"{test_metrics['ppg_channel']['pearson_r']}"
    )

    print(
        f"RESP Pearson R    : "
        f"{test_metrics['respiration_channel']['pearson_r']}"
    )

    print("-" * 72)

    print(
        f"Metrics saved to  : "
        f"{metrics_path}"
    )

    print(
        f"Model saved to    : "
        f"{checkpoint_path}"
    )

    print("=" * 72)

    print("\nSTATUS: BIDMC CNN AUTOENCODER BASELINE COMPLETE")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()