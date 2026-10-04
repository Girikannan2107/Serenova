"""
SERENOVA — BIDMC TEMPORAL EMBEDDING BENCHMARK

Purpose
-------
Benchmark temporal structure in the already-trained BIDMC CNN embeddings.

IMPORTANT:
- The waveform CNN is FROZEN and is not retrained.
- This script uses existing [64,157] embeddings.
- BIDMC has no Serenova clinical target label, so this is a
  self-supervised latent reconstruction benchmark.
- Input embeddings are corrupted with temporal masking.
- Models reconstruct the original frozen embeddings.

Compared models
---------------
1. Global temporal pooling baseline
2. Temporal CNN
3. Small Transformer encoder

Outputs
-------
data/results/bidmc/
    temporal_embedding_benchmark.json
    temporal_embedding_benchmark.pt
"""

from __future__ import annotations

import json
import math
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

EMBED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "bidmc"
    / "embeddings"
)

RESULTS_DIR = (
    BASE_DIR
    / "data"
    / "results"
    / "bidmc"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_PATH = EMBED_DIR / "train_embeddings.npy"
VAL_PATH = EMBED_DIR / "val_embeddings.npy"
TEST_PATH = EMBED_DIR / "test_embeddings.npy"


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 64
EPOCHS = 25

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MASK_RATIO = 0.15

EARLY_STOPPING_PATIENCE = 6

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42) -> None:
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

def load_embeddings(path: Path) -> torch.Tensor:
    x = np.load(path, mmap_mode="r")

    if x.ndim != 3:
        raise ValueError(
            f"Expected [N,C,T], got {x.shape} from {path}"
        )

    if x.shape[1] != 64 or x.shape[2] != 157:
        raise ValueError(
            f"Expected [N,64,157], got {x.shape} from {path}"
        )

    if not np.isfinite(x).all():
        raise ValueError(
            f"Non-finite values found in {path}"
        )

    # Explicit copy avoids non-writable NumPy tensor warnings.
    x = np.array(x, dtype=np.float32, copy=True)

    return torch.from_numpy(x)


# ============================================================
# TEMPORAL MASKING
# ============================================================

def make_masked_input(
    x: torch.Tensor,
    mask_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    x:
        [B,C,T]

    Returns
    -------
    corrupted:
        [B,C,T]

    mask:
        [B,1,T]

    A temporal mask is shared across channels so that the model
    must infer missing temporal information rather than individual
    channel values independently.
    """

    b, c, t = x.shape

    mask = (
        torch.rand(
            b,
            1,
            t,
            device=x.device,
        )
        < mask_ratio
    )

    corrupted = x.clone()

    # Zero masked temporal positions.
    corrupted = corrupted.masked_fill(mask, 0.0)

    return corrupted, mask


# ============================================================
# MODEL 1 — GLOBAL POOLING BASELINE
# ============================================================

class GlobalPoolingBaseline(nn.Module):
    """
    Mean-pool time dimension, process global representation,
    then broadcast the result over time.

    This intentionally removes explicit temporal information.
    """

    def __init__(
        self,
        channels: int = 64,
        hidden: int = 128,
        time_steps: int = 157,
    ):
        super().__init__()

        self.time_steps = time_steps

        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T]

        pooled = x.mean(dim=2)

        representation = self.mlp(pooled)

        output = representation.unsqueeze(-1)

        output = output.expand(
            -1,
            -1,
            self.time_steps,
        )

        return output


# ============================================================
# MODEL 2 — TEMPORAL CNN
# ============================================================

class TemporalCNN(nn.Module):
    """
    Lightweight temporal convolutional reconstruction head.

    Input:
        [B,64,157]

    Output:
        [B,64,157]
    """

    def __init__(
        self,
        channels: int = 64,
        hidden: int = 128,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                hidden,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(hidden),
            nn.GELU(),

            nn.Conv1d(
                hidden,
                hidden,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(hidden),
            nn.GELU(),

            nn.Conv1d(
                hidden,
                channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        output = self.net(x)

        return output + residual


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_len: int = 157,
    ):
        super().__init__()

        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(
            max_len,
            d_model,
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ============================================================
# MODEL 3 — SMALL TRANSFORMER
# ============================================================

class SmallTransformer(nn.Module):
    """
    Small temporal Transformer.

    Input:
        [B,64,157]

    Internally:
        [B,157,64]

    Output:
        [B,64,157]
    """

    def __init__(
        self,
        channels: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.positional_encoding = PositionalEncoding(
            d_model=channels,
            max_len=157,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output = nn.Linear(
            channels,
            channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # [B,C,T] -> [B,T,C]
        x = x.transpose(1, 2)

        x = self.positional_encoding(x)

        x = self.encoder(x)

        x = self.output(x)

        # [B,T,C] -> [B,C,T]
        x = x.transpose(1, 2)

        return x


# ============================================================
# MODEL FACTORY
# ============================================================

def build_model(name: str) -> nn.Module:

    if name == "global_pooling":
        return GlobalPoolingBaseline()

    if name == "temporal_cnn":
        return TemporalCNN()

    if name == "transformer":
        return SmallTransformer()

    raise ValueError(f"Unknown model: {name}")


# ============================================================
# METRICS
# ============================================================

def mse_metric(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:

    return float(
        torch.mean(
            (prediction - target) ** 2
        ).item()
    )


def masked_mse_metric(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:

    diff = (
        prediction - target
    ) ** 2

    expanded_mask = mask.expand_as(diff)

    selected = diff[expanded_mask]

    if selected.numel() == 0:
        return float("nan")

    return float(
        selected.mean().item()
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mask_ratio: float,
) -> dict:

    model.eval()

    total_mse = 0.0
    total_masked_mse = 0.0
    batches = 0

    for (target,) in loader:

        target = target.to(
            device,
            non_blocking=True,
        )

        corrupted, mask = make_masked_input(
            target,
            mask_ratio,
        )

        prediction = model(corrupted)

        total_mse += mse_metric(
            prediction,
            target,
        )

        total_masked_mse += masked_mse_metric(
            prediction,
            target,
            mask,
        )

        batches += 1

    return {
        "mse": total_mse / batches,
        "masked_mse": total_masked_mse / batches,
    }


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:

    print()
    print("=" * 72)
    print(f"MODEL: {name}")
    print("=" * 72)

    model = build_model(name).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: {parameter_count:,}"
    )

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

    best_val = float("inf")
    best_state = None
    best_epoch = 0

    history = []

    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):

        model.train()

        train_loss = 0.0
        train_masked_loss = 0.0
        batches = 0

        for (target,) in train_loader:

            target = target.to(
                device,
                non_blocking=True,
            )

            corrupted, mask = make_masked_input(
                target,
                MASK_RATIO,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = model(corrupted)

            loss = torch.mean(
                (prediction - target) ** 2
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            train_loss += loss.item()

            train_masked_loss += masked_mse_metric(
                prediction.detach(),
                target,
                mask,
            )

            batches += 1

        train_loss /= batches
        train_masked_loss /= batches

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            MASK_RATIO,
        )

        val_loss = val_metrics["mse"]

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_mse": train_loss,
            "train_masked_mse": train_masked_loss,
            "val_mse": val_loss,
            "val_masked_mse": val_metrics["masked_mse"],
            "learning_rate": current_lr,
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train={train_loss:.6f} | "
            f"train_masked={train_masked_loss:.6f} | "
            f"val={val_loss:.6f} | "
            f"val_masked={val_metrics['masked_mse']:.6f} | "
            f"lr={current_lr:.2e}"
        )

        if val_loss < best_val:

            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        else:

            patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(
                    "Early stopping."
                )
                break

    elapsed = time.time() - start_time

    if best_state is None:
        raise RuntimeError(
            f"No checkpoint captured for {name}"
        )

    model.load_state_dict(
        best_state
    )

    return {
        "model": model,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "training_seconds": elapsed,
        "history": history,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 72)
    print("SERENOVA — BIDMC TEMPORAL EMBEDDING BENCHMARK")
    print("=" * 72)

    print()
    print(f"Device: {DEVICE}")
    print(f"Embedding directory: {EMBED_DIR}")
    print(f"Mask ratio: {MASK_RATIO}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print()
    print("Loading frozen embeddings...")

    X_train = load_embeddings(
        TRAIN_PATH
    )

    X_val = load_embeddings(
        VAL_PATH
    )

    X_test = load_embeddings(
        TEST_PATH
    )

    print(
        f"Train: {tuple(X_train.shape)}"
    )

    print(
        f"Val  : {tuple(X_val.shape)}"
    )

    print(
        f"Test : {tuple(X_test.shape)}"
    )

    # --------------------------------------------------------
    # DATALOADERS
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

    generator = torch.Generator()

    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
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
    # BASELINE + TEMPORAL MODELS
    # --------------------------------------------------------

    model_names = [
        "global_pooling",
        "temporal_cnn",
        "transformer",
    ]

    results = {}

    checkpoints = {}

    for name in model_names:

        set_seed(SEED)

        result = train_model(
            name,
            train_loader,
            val_loader,
            DEVICE,
        )

        model = result["model"]

        test_metrics = evaluate(
            model,
            test_loader,
            DEVICE,
            MASK_RATIO,
        )

        print()
        print(
            f"{name} TEST MSE: "
            f"{test_metrics['mse']:.6f}"
        )

        print(
            f"{name} TEST masked MSE: "
            f"{test_metrics['masked_mse']:.6f}"
        )

        results[name] = {
            "parameter_count": result[
                "parameter_count"
            ],
            "best_epoch": result[
                "best_epoch"
            ],
            "best_val_mse": result[
                "best_val_mse"
            ],
            "test_mse": test_metrics[
                "mse"
            ],
            "test_masked_mse": test_metrics[
                "masked_mse"
            ],
            "training_seconds": result[
                "training_seconds"
            ],
            "history": result[
                "history"
            ],
        }

        checkpoints[name] = {
            key: value.cpu()
            for key, value in model.state_dict().items()
        }

    # --------------------------------------------------------
    # RANK MODELS
    # --------------------------------------------------------

    ranking = sorted(
        results.items(),
        key=lambda item: item[1]["test_masked_mse"],
    )

    ranking_names = [
        name
        for name, _ in ranking
    ]

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    output_json = (
        RESULTS_DIR
        / "temporal_embedding_benchmark.json"
    )

    output_pt = (
        RESULTS_DIR
        / "temporal_embedding_benchmark.pt"
    )

    report = {
        "dataset": "bidmc",
        "task": "self_supervised_temporal_latent_reconstruction",
        "seed": SEED,
        "device": str(DEVICE),
        "input_shape": [64, 157],
        "mask_ratio": MASK_RATIO,
        "batch_size": BATCH_SIZE,
        "epochs_max": EPOCHS,
        "frozen_waveform_encoder": True,
        "source_embeddings": {
            "train": str(TRAIN_PATH),
            "val": str(VAL_PATH),
            "test": str(TEST_PATH),
        },
        "models": results,
        "ranking_by_test_masked_mse": ranking_names,
    }

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=4,
        )

    torch.save(
        checkpoints,
        output_pt,
    )

    print()
    print("=" * 72)
    print("FINAL BENCHMARK")
    print("=" * 72)

    print()

    for rank, name in enumerate(
        ranking_names,
        start=1,
    ):

        metrics = results[name]

        print(
            f"{rank}. {name:18s} | "
            f"params={metrics['parameter_count']:>8,} | "
            f"val_masked={metrics['best_val_mse']:.6f} | "
            f"test_masked={metrics['test_masked_mse']:.6f}"
        )

    print()
    print("=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)

    print()
    print("Saved:")
    print(output_json)
    print(output_pt)


if __name__ == "__main__":
    main()