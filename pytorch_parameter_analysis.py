#!/usr/bin/env python3
"""
PyTorch-based complementary analysis across multi-parameter drive-cycle data.

This module trains a lightweight autoencoder on normalized numeric parameters to
derive reconstruction-error scorecards per feature. It is designed for CPU-only
execution but leverages GPU automatically when available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, random_split

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = Dataset = random_split = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS = 12
DEFAULT_MAX_SAMPLES = 120_000
DEFAULT_HOLDOUT_FRACTION = 0.1
DEFAULT_BOTTLENECK = 32
DEFAULT_HIDDEN = 128
EPS = 1e-9
UPLIFT_PLOT_TOP_N = 20


class FeatureTensorDataset(Dataset):
    """Simple Dataset wrapper for (n_samples, n_features) tensors."""

    def __init__(self, data: np.ndarray):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed; cannot create dataset.")
        self.tensor = torch.from_numpy(data.astype(np.float32))

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensor[idx]


class ParameterAutoencoder(nn.Module):
    """Shallow autoencoder tailored for multi-parameter reconstruction."""

    def __init__(self, input_dim: int, hidden_dim: int = DEFAULT_HIDDEN, bottleneck_dim: int = DEFAULT_BOTTLENECK):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)


@dataclass
class PreparedData:
    features: List[str]
    data_matrix: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    medians: np.ndarray
    missing_fraction: np.ndarray


def torch_ready() -> bool:
    """Return True if PyTorch is available."""
    return TORCH_AVAILABLE


def _select_feature_columns(df: pd.DataFrame, min_non_null: int = 500, exclude: Optional[List[str]] = None) -> List[str]:
    """Heuristic to identify informative numeric columns for analysis."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return []

    exclude = set(exclude or [])
    patterns = ("timestamp", "time_", "elapsed", "index", "sample", "distance_along_route")
    selected: List[str] = []
    for col in numeric_cols:
        if col in exclude:
            continue
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in patterns):
            continue
        series = df[col]
        non_null = int(series.notna().sum())
        if non_null < min_non_null:
            continue
        if float(series.std(skipna=True) or 0.0) < EPS:
            continue
        selected.append(col)
    return selected


def _prepare_feature_matrix(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    random_seed: int = 42,
) -> PreparedData:
    """Construct a standardized matrix suitable for PyTorch training."""
    if feature_columns is None:
        feature_columns = _select_feature_columns(df)

    if not feature_columns:
        raise ValueError("No suitable numeric feature columns were found for PyTorch analysis.")

    feature_df = df[feature_columns].copy()

    missing_fraction = feature_df.isna().mean().to_numpy(dtype=float)
    medians = feature_df.median().to_numpy(dtype=float)
    for idx, col in enumerate(feature_columns):
        feature_df[col] = feature_df[col].fillna(medians[idx])

    if len(feature_df) > max_samples:
        feature_df = feature_df.sample(n=max_samples, random_state=random_seed)

    means = feature_df.mean().to_numpy(dtype=float)
    stds = feature_df.std().replace(0, 1.0).to_numpy(dtype=float)

    normalized = (feature_df.to_numpy(dtype=float) - means) / stds

    return PreparedData(
        features=list(feature_columns),
        data_matrix=normalized,
        means=means,
        stds=stds,
        medians=medians,
        missing_fraction=missing_fraction,
    )


def _train_autoencoder(
    data: np.ndarray,
    input_dim: int,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    learning_rate: float = 1e-3,
    use_cuda: bool = True,
    random_seed: int = 42,
) -> Tuple[ParameterAutoencoder, Dict[str, List[float]], torch.device, int, int]:
    """Train an autoencoder on the provided data matrix."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed. Install torch to enable parameter analysis.")

    torch.manual_seed(random_seed)
    device = torch.device("cuda") if (use_cuda and torch.cuda.is_available()) else torch.device("cpu")

    dataset = FeatureTensorDataset(data)
    n_samples = len(dataset)
    if n_samples < batch_size:
        batch_size = max(32, n_samples // 2 or 1)

    holdout_size = int(n_samples * holdout_fraction)
    if holdout_size < 1:
        holdout_size = 1
    train_size = n_samples - holdout_size
    if train_size < 1:
        raise ValueError("Not enough samples for training after applying holdout split.")

    train_dataset, val_dataset = random_split(dataset, [train_size, holdout_size])
    if hasattr(train_dataset, "indices"):
        train_indices = list(train_dataset.indices)
    else:
        train_indices = list(range(train_size))
    if hasattr(val_dataset, "indices"):
        val_indices = list(val_dataset.indices)
    else:
        val_indices = list(range(train_size, train_size + holdout_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ParameterAutoencoder(input_dim=input_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item() * batch.size(0)

        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                val_loss = criterion(recon, batch)
                val_loss_accum += val_loss.item() * batch.size(0)

        train_epoch_loss = train_loss_accum / train_size
        val_epoch_loss = val_loss_accum / holdout_size
        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 4) == 0:
            print(f"    Epoch {epoch:02d}/{epochs} - train loss: {train_epoch_loss:.6f}, val loss: {val_epoch_loss:.6f}")

    return model, history, device, train_size, holdout_size, train_indices, val_indices


def _compute_feature_errors(
    model: ParameterAutoencoder,
    data: np.ndarray,
    *,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Compute per-feature absolute reconstruction error in original units."""
    model.eval()
    tensor = torch.from_numpy(data.astype(np.float32)).to(device)
    with torch.no_grad():
        recon = model(tensor).cpu().numpy()
    errors = np.abs(data - recon)
    mae_normalized = errors.mean(axis=0)
    mae_raw = mae_normalized * stds
    return mae_raw


def run_pytorch_parameter_analysis(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    timestamp: str,
    feature_columns: Optional[List[str]] = None,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = 1e-3,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    use_cuda: bool = False,
    random_seed: int = 42,
) -> Dict[str, object]:
    """
    Train an autoencoder across numeric parameters and generate complementary metrics.

    Returns a dictionary with training diagnostics, per-parameter reconstruction scores,
    and file paths to exported CSV/JSON artifacts.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed. Install torch to enable parameter analysis.")

    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = _prepare_feature_matrix(
        df,
        feature_columns=feature_columns,
        max_samples=max_samples,
        random_seed=random_seed,
    )

    print(
        f"  Training PyTorch autoencoder on {prepared.data_matrix.shape[0]} samples "
        f"with {len(prepared.features)} parameters (device={'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'})."
    )

    model, history, device, train_size, val_size, train_indices, val_indices = _train_autoencoder(
        prepared.data_matrix,
        input_dim=len(prepared.features),
        batch_size=batch_size,
        epochs=epochs,
        holdout_fraction=holdout_fraction,
        learning_rate=learning_rate,
        use_cuda=use_cuda,
        random_seed=random_seed,
    )

    mae_raw = _compute_feature_errors(
        model,
        prepared.data_matrix,
        means=prepared.means,
        stds=prepared.stds,
        device=device,
    )
    mae_pct_std = mae_raw / (prepared.stds + EPS)

    if val_indices:
        val_matrix = prepared.data_matrix[np.array(val_indices, dtype=int)]
        mae_val_raw = _compute_feature_errors(
            model,
            val_matrix,
            means=prepared.means,
            stds=prepared.stds,
            device=device,
        )
    else:
        mae_val_raw = mae_raw.copy()
    mae_val_pct_std = mae_val_raw / (prepared.stds + EPS)

    feature_summary = pd.DataFrame(
        {
            "feature": prepared.features,
            "mean": prepared.means,
            "std": prepared.stds,
            "median": prepared.medians,
            "missing_fraction": prepared.missing_fraction,
            "mae_raw": mae_raw,
            "mae_pct_std": mae_pct_std,
            "mae_val_raw": mae_val_raw,
            "mae_val_pct_std": mae_val_pct_std,
        }
    )
    feature_summary.sort_values(by="mae_pct_std", ascending=False, inplace=True)

    metrics_path = output_dir / f"pytorch_parameter_metrics_{timestamp}.csv"
    feature_summary.to_csv(metrics_path, index=False)

    model_config = {
        "input_dim": len(prepared.features),
        "hidden_dim": DEFAULT_HIDDEN,
        "bottleneck_dim": DEFAULT_BOTTLENECK,
    }

    model_path = output_dir / f"pytorch_autoencoder_{timestamp}.pt"
    if TORCH_AVAILABLE:
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": model_config,
            },
            model_path,
        )

    scaler_payload = {
        "features": prepared.features,
        "means": prepared.means.tolist(),
        "stds": prepared.stds.tolist(),
        "medians": prepared.medians.tolist(),
        "missing_fraction": prepared.missing_fraction.tolist(),
    }
    scaler_path = output_dir / f"pytorch_autoencoder_scalers_{timestamp}.json"
    with open(scaler_path, "w") as f_scaler:
        json.dump(scaler_payload, f_scaler, indent=2)

    val_mae_payload = {
        "features": prepared.features,
        "mae_raw": mae_val_raw.tolist(),
        "mae_pct_std": mae_val_pct_std.tolist(),
    }
    val_mae_path = output_dir / f"pytorch_real_val_mae_{timestamp}.json"
    with open(val_mae_path, "w") as f_val:
        json.dump(val_mae_payload, f_val, indent=2)

    top_parameters_serialized = [
        {key: _to_native(value) for key, value in record.items()}
        for record in feature_summary.head(10).to_dict(orient="records")
    ]

    artifact = {
        "timestamp": timestamp,
        "n_features": len(prepared.features),
        "n_samples_used": int(prepared.data_matrix.shape[0]),
        "train_samples": train_size,
        "val_samples": val_size,
        "device": str(device),
        "epochs": epochs,
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "metrics_csv": str(metrics_path),
        "top_parameters": top_parameters_serialized,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "real_val_mae_path": str(val_mae_path),
        "model_config": model_config,
    }

    json_path = output_dir / f"pytorch_parameter_metrics_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(artifact, f, indent=2)

    artifact["metrics_json"] = str(json_path)
    print(f"  Saved PyTorch parameter scorecard: {metrics_path.name}")

    return artifact


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    with open(path) as f:
        return json.load(f)


def _to_native(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _prepare_dataset_for_autoencoder(
    df: pd.DataFrame,
    features: List[str],
    medians: np.ndarray,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Standard dataset is empty; cannot compute uplift.")

    coerced = {}
    for idx, feature in enumerate(features):
        if feature in df.columns:
            series = pd.to_numeric(df[feature], errors="coerce")
            coerced[feature] = series.fillna(medians[idx])
        else:
            coerced[feature] = pd.Series(medians[idx], index=df.index)
    return pd.DataFrame(coerced, index=df.index, columns=features)


def run_standard_cycle_uplift(
    standard_df: pd.DataFrame,
    artifact: Dict[str, object],
    *,
    output_dir: Path,
    timestamp: str,
    top_n: int = UPLIFT_PLOT_TOP_N,
) -> Dict[str, object]:
    """
    Evaluate the trained autoencoder on standard-cycle data and compute per-parameter uplift.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed; cannot compute uplift.")

    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(artifact.get("model_path", ""))
    scaler_path = Path(artifact.get("scaler_path", ""))
    val_mae_path = Path(artifact.get("real_val_mae_path", ""))
    model_config = artifact.get("model_config") or {}

    if not model_path.exists():
        raise FileNotFoundError(f"Autoencoder weights not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler bundle not found: {scaler_path}")
    if not val_mae_path.exists():
        raise FileNotFoundError(f"Validation MAE bundle not found: {val_mae_path}")

    scaler_info = _load_json(scaler_path)
    features = scaler_info["features"]
    means = np.asarray(scaler_info["means"], dtype=float)
    stds = np.asarray(scaler_info["stds"], dtype=float)
    medians = np.asarray(scaler_info["medians"], dtype=float)
    stds_safe = np.where(np.abs(stds) < EPS, 1.0, stds)

    standard_features_df = _prepare_dataset_for_autoencoder(standard_df, features, medians)
    if standard_features_df.empty:
        raise ValueError("Prepared standard feature matrix is empty.")

    normalized = (standard_features_df.to_numpy(dtype=float) - means) / stds_safe

    hidden_dim = int(model_config.get("hidden_dim", DEFAULT_HIDDEN))
    bottleneck_dim = int(model_config.get("bottleneck_dim", DEFAULT_BOTTLENECK))
    model = ParameterAutoencoder(len(features), hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)
    state_dict = torch.load(model_path, map_location="cpu")
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    model.load_state_dict(state_dict)
    model.to(torch.device("cpu"))
    model.eval()

    tensor = torch.from_numpy(normalized.astype(np.float32))
    with torch.no_grad():
        recon = model(tensor).cpu().numpy()

    errors = np.abs(normalized - recon)
    mae_std_normalized = errors.mean(axis=0)
    mae_std_raw = mae_std_normalized * stds_safe
    mae_std_pct_std = mae_std_raw / stds_safe

    val_mae_info = _load_json(val_mae_path)
    val_map = {feat: float(val) for feat, val in zip(val_mae_info.get("features", []), val_mae_info.get("mae_raw", []))}
    mae_val_raw = np.array([val_map.get(feat, 0.0) for feat in features], dtype=float)
    mae_val_pct_std = mae_val_raw / stds_safe

    uplift = (mae_std_raw - mae_val_raw) / stds_safe

    uplift_df = pd.DataFrame(
        {
            "feature": features,
            "std_real": stds_safe,
            "mae_real_val": mae_val_raw,
            "mae_real_val_pct_std": mae_val_pct_std,
            "mae_standard": mae_std_raw,
            "mae_standard_pct_std": mae_std_pct_std,
            "uplift": uplift,
        }
    ).sort_values(by="uplift", ascending=False)

    csv_path = output_dir / f"pytorch_standard_uplift_{timestamp}.csv"
    json_path = output_dir / f"pytorch_standard_uplift_{timestamp}.json"
    uplift_df.to_csv(csv_path, index=False)
    serializable_records = [
        {key: _to_native(value) for key, value in record.items()}
        for record in uplift_df.to_dict(orient="records")
    ]
    with open(json_path, "w") as f_json:
        json.dump({"timestamp": timestamp, "records": serializable_records}, f_json, indent=2)

    plot_path: Optional[Path] = None
    if MATPLOTLIB_AVAILABLE and not uplift_df.empty:
        plot_path = output_dir / f"pytorch_standard_uplift_{timestamp}.png"
        top_plot = uplift_df.head(max(5, min(top_n, len(uplift_df))))
        plt.figure(figsize=(10, max(6, top_plot.shape[0] * 0.4)))
        plt.barh(top_plot["feature"][::-1], top_plot["uplift"][::-1], color="#1f77b4")
        plt.xlabel("Uplift (delta-MAE / std)")
        plt.ylabel("Feature")
        plt.title("Standard Cycle Uplift (delta-MAE/std)")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()

    top_features = [
        {"feature": record["feature"], "uplift": _to_native(record["uplift"])}
        for record in uplift_df.head(10)[["feature", "uplift"]].to_dict(orient="records")
    ]

    return {
        "uplift_csv": str(csv_path),
        "uplift_json": str(json_path),
        "uplift_plot": str(plot_path) if plot_path else None,
        "top_features": top_features,
        "n_samples": int(standard_features_df.shape[0]),
        "n_features": len(features),
    }


__all__ = [
    "TORCH_AVAILABLE",
    "run_pytorch_parameter_analysis",
    "run_standard_cycle_uplift",
    "torch_ready",
]
