
from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/processed/model_sequences")
MODEL_DIR = Path("models/ablation")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 256
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
MAX_EPOCHS = 20
PATIENCE = 5
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

POLLUTANT_FEATURES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]

TIME_FEATURES = [
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "day_of_week_sin", "day_of_week_cos",
]

LOCATION_FEATURES = ["Latitude", "Longitude"]

WEATHER_FEATURES = [
    "temperature_C",
    "dewpoint_C",
    "relative_humidity",
    "wind_speed",
    "precipitation_mm",
    "wind_direction_sin",
    "wind_direction_cos",
]

TARGET_NAMES = ["PM2.5", "PM10"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET: SELECT FEATURES WITHOUT COPYING ALL ARRAYS
# ============================================================

class FeatureDataset(Dataset):
    def __init__(self, x_path, y_path, feature_indices):
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.feature_indices = np.asarray(
            feature_indices, dtype=np.int64
        )

        if len(self.X) != len(self.y):
            raise ValueError("Input and target counts differ.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Read only the selected feature columns.
        x = np.array(
            self.X[idx][:, self.feature_indices],
            dtype=np.float32,
            copy=True,
        )
        y = np.array(self.y[idx], dtype=np.float32, copy=True)

        return torch.from_numpy(x), torch.from_numpy(y)


class ScaledTargetDataset(Dataset):
    def __init__(self, base, target_mean, target_std):
        self.base = base
        self.target_mean = torch.tensor(
            target_mean, dtype=torch.float32
        )
        self.target_std = torch.tensor(
            target_std, dtype=torch.float32
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        y = (y - self.target_mean) / self.target_std
        return x, y


# ============================================================
# MODEL
# ============================================================

class LSTMForecaster(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )

        self.regressor = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        return self.regressor(sequence_output[:, -1, :])


# ============================================================
# METRICS IN ORIGINAL POLLUTANT UNITS
# ============================================================

def calculate_metrics(actual, predicted):
    results = {}

    for j, name in enumerate(TARGET_NAMES):
        error = predicted[:, j] - actual[:, j]

        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error ** 2))
        ss_res = np.sum(error ** 2)
        ss_tot = np.sum((actual[:, j] - actual[:, j].mean()) ** 2)

        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

        results[name] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2) if r2 is not None else None,
        }

    return results


def evaluate(model, loader, target_mean, target_std):
    model.eval()

    total_loss = 0.0
    count = 0
    predictions = []
    actuals = []

    criterion = nn.MSELoss()

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            pred = model(xb)
            loss = criterion(pred, yb)

            total_loss += loss.item() * len(xb)
            count += len(xb)

            pred_raw = (
                pred.cpu().numpy() * target_std + target_mean
            )
            actual_raw = (
                yb.cpu().numpy() * target_std + target_mean
            )

            predictions.append(pred_raw)
            actuals.append(actual_raw)

    return (
        total_loss / count,
        np.concatenate(predictions),
        np.concatenate(actuals),
    )


# ============================================================
# TRAIN ONE EXPERIMENT
# ============================================================

def run_experiment(
    experiment_name,
    feature_names,
    feature_indices,
    target_mean,
    target_std,
):
    print("\n" + "=" * 72)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 72)

    seed_everything(SEED)

    train_base = FeatureDataset(
        DATA_DIR / "X_train.npy",
        DATA_DIR / "y_train.npy",
        feature_indices,
    )
    val_base = FeatureDataset(
        DATA_DIR / "X_validation.npy",
        DATA_DIR / "y_validation.npy",
        feature_indices,
    )
    test_base = FeatureDataset(
        DATA_DIR / "X_test.npy",
        DATA_DIR / "y_test.npy",
        feature_indices,
    )

    train_ds = ScaledTargetDataset(
        train_base, target_mean, target_std
    )
    val_ds = ScaledTargetDataset(
        val_base, target_mean, target_std
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    test_loader = DataLoader(
        test_base,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = LSTMForecaster(
        input_size=len(feature_indices)
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_state = None
    patience_count = 0
    history = []

    checkpoint_path = MODEL_DIR / f"{experiment_name}_best.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            pred = model(xb)
            loss = criterion(pred, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            train_loss_sum += loss.item() * len(xb)
            train_count += len(xb)

        train_loss = train_loss_sum / train_count

        val_loss, val_pred, val_actual = evaluate(
            model, val_loader, target_mean, target_std
        )
        val_metrics = calculate_metrics(val_actual, val_pred)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train MSE={train_loss:.5f} | "
            f"Validation MSE={val_loss:.5f}"
        )
        print(
            f"  PM2.5 MAE={val_metrics['PM2.5']['MAE']:.3f} | "
            f"PM10 MAE={val_metrics['PM10']['MAE']:.3f}"
        )

        history.append({
            "epoch": epoch,
            "train_mse_scaled": train_loss,
            "validation_mse_scaled": val_loss,
            "validation_PM25_MAE": val_metrics["PM2.5"]["MAE"],
            "validation_PM10_MAE": val_metrics["PM10"]["MAE"],
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0

            torch.save(
                {
                    "model_state_dict": best_state,
                    "input_size": len(feature_indices),
                    "feature_names": feature_names,
                    "target_mean": target_mean.tolist(),
                    "target_std": target_std.tolist(),
                    "seed": SEED,
                },
                checkpoint_path,
            )
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint created for {experiment_name}")

    model.load_state_dict(best_state)

    # Test set is evaluated only after validation-based model selection.
    _, test_predictions, test_actuals = evaluate(
        model, test_loader, target_mean, target_std
    )

    test_metrics = calculate_metrics(
        test_actuals, test_predictions
    )

    np.save(
        MODEL_DIR / f"{experiment_name}_test_predictions.npy",
        test_predictions.astype(np.float32),
    )

    pd.DataFrame(history).to_csv(
        MODEL_DIR / f"{experiment_name}_history.csv",
        index=False,
    )

    print(f"\n{experiment_name} FINAL TEST RESULTS")

    for target, result in test_metrics.items():
        print(
            f"{target}: MAE={result['MAE']:.4f}, "
            f"RMSE={result['RMSE']:.4f}, "
            f"R2={result['R2']:.4f}"
        )

    return test_metrics


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("POLLUTION-ONLY VS POLLUTION + WEATHER ABLATION")
    print("=" * 72)
    print("Device:", DEVICE)

    metadata_path = DATA_DIR / "preprocessing_metadata.json"

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    all_features = metadata["feature_names"]

    required_baseline = (
        POLLUTANT_FEATURES + TIME_FEATURES + LOCATION_FEATURES
    )
    required_full = required_baseline + WEATHER_FEATURES

    for name in required_full:
        if name not in all_features:
            raise ValueError(
                f"Feature {name!r} missing from metadata. "
                "Inspect preprocessing_metadata.json."
            )

    baseline_indices = [
        all_features.index(name) for name in required_baseline
    ]
    full_indices = [
        all_features.index(name) for name in required_full
    ]

    # Fit target scaling using training targets only.
    y_train = np.load(
        DATA_DIR / "y_train.npy", mmap_mode="r"
    )
    target_mean = np.asarray(y_train, dtype=np.float64).mean(axis=0)
    target_std = np.asarray(y_train, dtype=np.float64).std(axis=0)
    target_std[target_std < 1e-8] = 1.0
    del y_train

    print(f"\nPollution + time + location: {len(baseline_indices)} features")
    print(f"Pollution + weather + time + location: {len(full_indices)} features")
    print("Targets: next-hour PM2.5 and PM10")
    print("Train/validation/test split remains unchanged.")

    baseline_results = run_experiment(
        "pollution_only",
        required_baseline,
        baseline_indices,
        target_mean,
        target_std,
    )

    full_results = run_experiment(
        "pollution_weather",
        required_full,
        full_indices,
        target_mean,
        target_std,
    )

    # Positive percentage means the weather model reduced the error.
    comparison = {}

    print("\n" + "=" * 72)
    print("FINAL ABLATION COMPARISON")
    print("=" * 72)

    for target in TARGET_NAMES:
        comparison[target] = {}

        print(f"\n{target}")
        for metric in ["MAE", "RMSE"]:
            baseline_value = baseline_results[target][metric]
            full_value = full_results[target][metric]

            improvement = (
                (baseline_value - full_value) / baseline_value * 100
                if baseline_value != 0 else 0.0
            )

            comparison[target][metric] = {
                "pollution_only": baseline_value,
                "pollution_weather": full_value,
                "weather_error_reduction_percent": float(improvement),
            }

            print(
                f"{metric}: without weather={baseline_value:.4f} | "
                f"with weather={full_value:.4f} | "
                f"error reduction={improvement:.2f}%"
            )

        comparison[target]["R2"] = {
            "pollution_only": baseline_results[target]["R2"],
            "pollution_weather": full_results[target]["R2"],
        }

        print(
            f"R2: without weather={baseline_results[target]['R2']:.4f} | "
            f"with weather={full_results[target]['R2']:.4f}"
        )

    output = MODEL_DIR / "ablation_comparison.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved to: {output}")
    print("Ablation experiments completed.")


if __name__ == "__main__":
    main()
