
"""
Multi-seed robustness experiment for air-quality forecasting.

Experiments:
1. pollution_only
2. pollution_weather

Seeds: 21, 42, 123

The script:
- Preserves the existing LSTM architecture.
- Uses validation loss for checkpoint selection and early stopping.
- Evaluates the test set with correctly scaled targets.
- Saves checkpoints separately for each seed and experiment.
- Reports mean and standard deviation across seeds.

Existing checkpoints are NOT overwritten.
"""

from pathlib import Path
import copy
import csv
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "processed" / "model_sequences"
EXISTING_MODEL_DIR = ROOT / "models" / "ablation"
OUTPUT_DIR = EXISTING_MODEL_DIR / "multiseed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [21, 42, 123]

BATCH_SIZE = 256
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
MAX_EPOCHS = 20
PATIENCE = 5

TARGET_NAMES = ["PM2.5", "PM10"]

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


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# 3. DATASETS
# ============================================================

class FeatureDataset(Dataset):
    """
    Select a subset of the existing 27 input features.

    y remains in original pollutant units unless wrapped
    with ScaledTargetDataset.
    """

    def __init__(self, x_path, y_path, feature_indices):
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

        self.feature_indices = np.asarray(
            feature_indices, dtype=np.int64
        )

        if len(self.X) != len(self.y):
            raise ValueError(
                f"Input/target count mismatch: "
                f"{len(self.X)} vs {len(self.y)}"
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = np.array(
            self.X[idx][:, self.feature_indices],
            dtype=np.float32,
            copy=True,
        )

        y = np.array(
            self.y[idx],
            dtype=np.float32,
            copy=True,
        )

        return torch.from_numpy(x), torch.from_numpy(y)


class ScaledTargetDataset(Dataset):
    """Scale target labels using training-set statistics."""

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
        y_scaled = (y - self.target_mean) / self.target_std

        return x, y_scaled


# ============================================================
# 4. MODEL
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
# 5. METRICS
# ============================================================

def calculate_metrics(actual, predicted):
    results = {}

    for j, name in enumerate(TARGET_NAMES):
        actual_column = actual[:, j]
        predicted_column = predicted[:, j]

        error = predicted_column - actual_column

        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error ** 2))

        ss_res = np.sum(error ** 2)
        ss_tot = np.sum(
            (actual_column - actual_column.mean()) ** 2
        )

        r2 = (
            1.0 - ss_res / ss_tot
            if ss_tot > 0
            else None
        )

        results[name] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2) if r2 is not None else None,
        }

    return results


def evaluate_scaled_targets(
    model,
    loader,
    target_mean,
    target_std,
):
    """
    Validation evaluation.

    Loader targets are scaled.
    Loss is computed in scaled target space.
    Metrics are calculated in original units.
    """

    model.eval()

    total_loss = 0.0
    count = 0

    predictions = []
    actuals = []

    criterion = nn.MSELoss()

    mean_tensor = torch.tensor(
        target_mean, dtype=torch.float32, device=DEVICE
    )
    std_tensor = torch.tensor(
        target_std, dtype=torch.float32, device=DEVICE
    )

    with torch.no_grad():
        for xb, yb_scaled in loader:
            xb = xb.to(DEVICE)
            yb_scaled = yb_scaled.to(DEVICE)

            pred_scaled = model(xb)

            loss = criterion(pred_scaled, yb_scaled)

            total_loss += loss.item() * len(xb)
            count += len(xb)

            pred_raw = (
                pred_scaled * std_tensor + mean_tensor
            )

            actual_raw = (
                yb_scaled * std_tensor + mean_tensor
            )

            predictions.append(
                pred_raw.cpu().numpy()
            )
            actuals.append(
                actual_raw.cpu().numpy()
            )

    return (
        total_loss / count,
        np.concatenate(predictions),
        np.concatenate(actuals),
    )


def evaluate_raw_test_targets(
    model,
    loader,
    target_mean,
    target_std,
):
    """
    Test evaluation.

    Test targets are already in original units.
    Inverse-transform predictions only.
    Never inverse-transform actual test targets.
    """

    model.eval()

    predictions = []
    actuals = []

    mean_array = np.asarray(
        target_mean, dtype=np.float32
    ).reshape(1, 2)

    std_array = np.asarray(
        target_std, dtype=np.float32
    ).reshape(1, 2)

    with torch.no_grad():
        for xb, yb_raw in loader:
            xb = xb.to(DEVICE)

            pred_scaled = model(xb).cpu().numpy()

            pred_raw = pred_scaled * std_array + mean_array

            predictions.append(pred_raw)
            actuals.append(yb_raw.numpy())

    return (
        np.concatenate(predictions),
        np.concatenate(actuals),
    )


# ============================================================
# 6. LOAD FEATURE DEFINITIONS AND TARGET SCALING
# ============================================================

def load_configuration():
    metadata_path = DATA_DIR / "preprocessing_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    all_features = metadata.get("feature_names")

    if not all_features:
        raise ValueError(
            "preprocessing_metadata.json does not contain "
            "'feature_names'."
        )

    if len(all_features) != 27:
        raise ValueError(
            f"Expected 27 features, found {len(all_features)}."
        )

    # Reuse the target-scaling statistics from the existing
    # experiment so the new runs remain directly comparable.
    reference_checkpoint = (
        EXISTING_MODEL_DIR / "pollution_only_best.pt"
    )

    if not reference_checkpoint.exists():
        raise FileNotFoundError(
            "Existing pollution_only_best.pt is required to "
            "reuse the original target scaling statistics."
        )

    try:
        checkpoint = torch.load(
            reference_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            reference_checkpoint,
            map_location="cpu",
        )

    target_mean = np.asarray(
        checkpoint["target_mean"], dtype=np.float32
    )

    target_std = np.asarray(
        checkpoint["target_std"], dtype=np.float32
    )

    if target_mean.shape != (2,) or target_std.shape != (2,):
        raise ValueError(
            "Expected two target means and two target standard "
            "deviations."
        )

    if np.any(target_std <= 0):
        raise ValueError("Target standard deviations must be positive.")

    # Verify that the saved feature names agree with the metadata.
    checkpoint_features = checkpoint["feature_names"]

    if len(checkpoint_features) != int(checkpoint["input_size"]):
        raise ValueError(
            "Reference checkpoint feature names and input size differ."
        )

    feature_lookup = {
        name: index for index, name in enumerate(all_features)
    }

    pollution_only_names = (
        POLLUTANT_FEATURES
        + TIME_FEATURES
        + LOCATION_FEATURES
    )

    pollution_weather_names = (
        pollution_only_names
        + WEATHER_FEATURES
    )

    for name in pollution_weather_names:
        if name not in feature_lookup:
            raise ValueError(
                f"Expected feature {name!r} is missing from metadata. "
                f"Available features: {all_features}"
            )

    experiments = {
        "pollution_only": pollution_only_names,
        "pollution_weather": pollution_weather_names,
    }

    return (
        all_features,
        experiments,
        target_mean,
        target_std,
    )


# ============================================================
# 7. TRAIN ONE EXPERIMENT FOR ONE SEED
# ============================================================

def run_experiment(
    experiment_name,
    feature_names,
    all_feature_names,
    target_mean,
    target_std,
    seed,
):
    print("\n" + "=" * 78)
    print(f"EXPERIMENT: {experiment_name} | SEED: {seed}")
    print("=" * 78)

    seed_everything(seed)

    feature_lookup = {
        name: index
        for index, name in enumerate(all_feature_names)
    }

    feature_indices = [
        feature_lookup[name]
        for name in feature_names
    ]

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

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        generator=generator,
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
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_count = 0
    history = []

    run_dir = OUTPUT_DIR / f"seed_{seed}" / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "best_model.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        loss_sum = 0.0
        sample_count = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            prediction = model(xb)
            loss = criterion(prediction, yb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            loss_sum += loss.item() * len(xb)
            sample_count += len(xb)

        train_loss = loss_sum / sample_count

        val_loss, val_predictions, val_actuals = (
            evaluate_scaled_targets(
                model,
                val_loader,
                target_mean,
                target_std,
            )
        )

        val_metrics = calculate_metrics(
            val_actuals,
            val_predictions,
        )

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train MSE={train_loss:.5f} | "
            f"Validation MSE={val_loss:.5f} | "
            f"PM2.5 MAE={val_metrics['PM2.5']['MAE']:.3f} | "
            f"PM10 MAE={val_metrics['PM10']['MAE']:.3f}"
        )

        history.append({
            "epoch": epoch,
            "train_mse_scaled": float(train_loss),
            "validation_mse_scaled": float(val_loss),
            "validation_PM25_MAE": val_metrics["PM2.5"]["MAE"],
            "validation_PM10_MAE": val_metrics["PM10"]["MAE"],
        })

        # Checkpoint selection uses validation loss only.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0

            torch.save(
                {
                    "model_state_dict": best_state,
                    "input_size": len(feature_indices),
                    "feature_names": feature_names,
                    "target_mean": target_mean.tolist(),
                    "target_std": target_std.tolist(),
                    "seed": seed,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "best_epoch": best_epoch,
                    "best_validation_loss": float(best_val_loss),
                },
                checkpoint_path,
            )

        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError(
            f"No best checkpoint created for {experiment_name}, "
            f"seed {seed}."
        )

    # Load the best validation-selected state before testing.
    model.load_state_dict(best_state)

    # Test set is evaluated once for this trained run.
    test_predictions, test_actuals = evaluate_raw_test_targets(
        model,
        test_loader,
        target_mean,
        target_std,
    )

    test_metrics = calculate_metrics(
        test_actuals,
        test_predictions,
    )

    predictions_path = run_dir / "test_predictions.npy"

    np.save(
        predictions_path,
        np.column_stack([
            test_actuals,
            test_predictions,
        ]),
    )

    with open(
        run_dir / "training_history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(history, f, indent=2)

    run_results = {
        "experiment": experiment_name,
        "seed": seed,
        "input_features": len(feature_names),
        "best_epoch": best_epoch,
        "best_validation_mse_scaled": float(best_val_loss),
        "test_metrics": test_metrics,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "predictions": str(predictions_path.relative_to(ROOT)),
    }

    with open(
        run_dir / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(run_results, f, indent=2)

    for pollutant in TARGET_NAMES:
        m = test_metrics[pollutant]
        print(
            f"TEST {pollutant}: "
            f"MAE={m['MAE']:.4f}, "
            f"RMSE={m['RMSE']:.4f}, "
            f"R2={m['R2']:.4f}"
        )

    print(f"Best epoch: {best_epoch}")
    print(f"Saved run to: {run_dir}")

    return run_results


# ============================================================
# 8. AGGREGATE RESULTS ACROSS SEEDS
# ============================================================

def summarize_results(all_results):
    summary = {}

    for experiment_name in [
        "pollution_only",
        "pollution_weather",
    ]:
        experiment_runs = [
            run for run in all_results
            if run["experiment"] == experiment_name
        ]

        summary[experiment_name] = {
            "number_of_seeds": len(experiment_runs),
            "seeds": [run["seed"] for run in experiment_runs],
            "metrics": {},
        }

        for pollutant in TARGET_NAMES:
            summary[experiment_name]["metrics"][pollutant] = {}

            for metric in ["MAE", "RMSE", "R2"]:
                values = np.asarray([
                    run["test_metrics"][pollutant][metric]
                    for run in experiment_runs
                ], dtype=float)

                summary[experiment_name]["metrics"][pollutant][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "individual_runs": values.tolist(),
                }

    return summary


def save_results(all_results, summary):
    raw_results_path = OUTPUT_DIR / "all_seed_results.json"

    with open(
        raw_results_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(all_results, f, indent=2)

    summary_path = OUTPUT_DIR / "multiseed_summary.json"

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    csv_path = OUTPUT_DIR / "multiseed_metrics.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "Experiment",
            "Seeds",
            "Pollutant",
            "Metric",
            "Mean",
            "Standard deviation",
        ])

        for experiment_name, experiment_data in summary.items():
            seeds_text = ";".join(
                str(seed) for seed in experiment_data["seeds"]
            )

            for pollutant, metrics in experiment_data["metrics"].items():
                for metric, values in metrics.items():
                    writer.writerow([
                        experiment_name,
                        seeds_text,
                        pollutant,
                        metric,
                        values["mean"],
                        values["std"],
                    ])

    print(f"\nSaved individual results: {raw_results_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved summary CSV: {csv_path}")


def print_summary(summary):
    print("\n" + "=" * 100)
    print("MULTI-SEED ROBUSTNESS SUMMARY")
    print("Mean +/- sample standard deviation across seeds")
    print("=" * 100)

    for experiment_name, experiment_data in summary.items():
        print(f"\n{experiment_name}")
        print(f"Seeds: {experiment_data['seeds']}")

        for pollutant, metrics in experiment_data["metrics"].items():
            print(f"\n  {pollutant}")

            for metric in ["MAE", "RMSE", "R2"]:
                values = metrics[metric]

                print(
                    f"    {metric}: "
                    f"{values['mean']:.4f} +/- {values['std']:.4f}"
                )

    print("\n" + "=" * 100)


# ============================================================
# 9. MAIN
# ============================================================

def main():
    print("MULTI-SEED ABLATION ROBUSTNESS EXPERIMENT")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"Output directory: {OUTPUT_DIR}")

    (
        all_features,
        experiments,
        target_mean,
        target_std,
    ) = load_configuration()

    print(f"\nNumber of available features: {len(all_features)}")
    print(f"Target means: {target_mean.tolist()}")
    print(f"Target standard deviations: {target_std.tolist()}")

    for name, features in experiments.items():
        print(f"{name}: {len(features)} features")

    all_results = []

    for seed in SEEDS:
        for experiment_name, feature_names in experiments.items():
            run_result = run_experiment(
                experiment_name=experiment_name,
                feature_names=feature_names,
                all_feature_names=all_features,
                target_mean=target_mean,
                target_std=target_std,
                seed=seed,
            )

            all_results.append(run_result)

    summary = summarize_results(all_results)

    save_results(all_results, summary)
    print_summary(summary)

    print("\nALL MULTI-SEED EXPERIMENTS COMPLETED.")


if __name__ == "__main__":
    main()
