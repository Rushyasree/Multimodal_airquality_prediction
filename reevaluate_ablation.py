
"""
Re-evaluate saved ablation checkpoints correctly.

Experiments:
1. pollution_only
2. pollution_weather

Important:
- X_test.npy contains preprocessed/scaled input features.
- y_test.npy contains RAW target values.
- Model outputs are scaled target predictions.
- Only predictions are inverse-transformed.
- Actual targets are NOT inverse-transformed again.
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

SEQUENCE_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "model_sequences"
)

ABLATION_DIR = ROOT_DIR / "models" / "ablation"

METADATA_PATH = SEQUENCE_DIR / "preprocessing_metadata.json"
X_TEST_PATH = SEQUENCE_DIR / "X_test.npy"
Y_TEST_PATH = SEQUENCE_DIR / "y_test.npy"

BATCH_SIZE = 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. MODEL ARCHITECTURE
# ============================================================

class LSTMForecaster(nn.Module):
    """
    Architecture used by train_ablation.py:

    LSTM
      -> last time-step output
      -> Linear(hidden_size, 64)
      -> ReLU
      -> Dropout
      -> Linear(64, 2)

    Outputs:
      scaled PM2.5 and PM10 predictions
    """

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        lstm_output, _ = self.lstm(x)

        # Use the output at the last historical time step.
        last_time_step = lstm_output[:, -1, :]

        return self.regressor(last_time_step)


# ============================================================
# 3. LOAD METADATA AND TEST ARRAYS
# ============================================================

def load_test_data():
    required_files = [
        METADATA_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_names = metadata.get("feature_names")

    if not feature_names:
        raise ValueError(
            "Could not find 'feature_names' in "
            f"{METADATA_PATH}"
        )

    X_test = np.load(X_TEST_PATH).astype(np.float32)
    y_test = np.load(Y_TEST_PATH).astype(np.float32)

    if X_test.ndim != 3:
        raise ValueError(
            f"Expected X_test with 3 dimensions, got {X_test.shape}"
        )

    if y_test.ndim != 2 or y_test.shape[1] != 2:
        raise ValueError(
            f"Expected y_test shape (samples, 2), got {y_test.shape}"
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test have different sample counts: "
            f"{len(X_test)} versus {len(y_test)}"
        )

    if X_test.shape[2] != len(feature_names):
        raise ValueError(
            "Feature count mismatch: X_test has "
            f"{X_test.shape[2]} features, but metadata lists "
            f"{len(feature_names)}."
        )

    if not np.isfinite(X_test).all():
        raise ValueError("X_test contains NaN or infinite values.")

    if not np.isfinite(y_test).all():
        raise ValueError("y_test contains NaN or infinite values.")

    print(f"Test input shape: {X_test.shape}")
    print(f"Test target shape: {y_test.shape}")
    print(f"Metadata feature count: {len(feature_names)}")

    return X_test, y_test, feature_names


# ============================================================
# 4. INFER ARCHITECTURE FROM CHECKPOINT
# ============================================================

def infer_architecture(checkpoint):
    state_dict = checkpoint["model_state_dict"]

    required_keys = [
        "lstm.weight_ih_l0",
        "lstm.weight_hh_l0",
        "regressor.0.weight",
        "regressor.3.weight",
    ]

    missing_keys = [
        key for key in required_keys if key not in state_dict
    ]

    if missing_keys:
        raise KeyError(
            "Checkpoint is missing expected model parameters: "
            f"{missing_keys}"
        )

    # PyTorch LSTM stores four gates for each hidden unit.
    hidden_size = state_dict["lstm.weight_hh_l0"].shape[1]

    # Count the saved LSTM input-weight matrices by layer number.
    layer_numbers = set()

    for key in state_dict:
        match = re.fullmatch(
            r"lstm\.weight_ih_l(\d+)",
            key,
        )

        if match:
            layer_numbers.add(int(match.group(1)))

    if not layer_numbers:
        raise ValueError(
            "Could not infer the number of LSTM layers."
        )

    num_layers = max(layer_numbers) + 1

    # Confirm that all layers from zero through num_layers - 1 exist.
    expected_layers = set(range(num_layers))

    if layer_numbers != expected_layers:
        raise ValueError(
            f"Unexpected LSTM layer indices: {sorted(layer_numbers)}"
        )

    input_size = int(checkpoint["input_size"])

    # Validate against the first layer's saved weight matrix.
    saved_input_size = state_dict["lstm.weight_ih_l0"].shape[1]

    if input_size != saved_input_size:
        raise ValueError(
            f"Checkpoint input_size={input_size}, but its saved "
            f"first-layer weights expect {saved_input_size}."
        )

    print(
        f"Inferred architecture: input_size={input_size}, "
        f"hidden_size={hidden_size}, "
        f"num_layers={num_layers}"
    )

    return input_size, hidden_size, num_layers


# ============================================================
# 5. EVALUATE ONE CHECKPOINT
# ============================================================

def evaluate_checkpoint(
    experiment,
    X_test,
    y_test,
    all_feature_names,
):
    checkpoint_path = ABLATION_DIR / f"{experiment}_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print("\n" + "=" * 70)
    print(f"Evaluating experiment: {experiment}")
    print("=" * 70)

    # Load the trusted, locally generated checkpoint.
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    required_checkpoint_keys = [
        "model_state_dict",
        "input_size",
        "feature_names",
        "target_mean",
        "target_std",
    ]

    missing = [
        key for key in required_checkpoint_keys
        if key not in checkpoint
    ]

    if missing:
        raise KeyError(
            f"{experiment} checkpoint is missing keys: {missing}"
        )

    saved_features = checkpoint["feature_names"]

    if len(saved_features) != int(checkpoint["input_size"]):
        raise ValueError(
            "Checkpoint input_size does not match its feature_names "
            f"for experiment {experiment}."
        )

    # Map each checkpoint feature to its position in X_test.
    feature_to_index = {
        name: index
        for index, name in enumerate(all_feature_names)
    }

    missing_features = [
        name for name in saved_features
        if name not in feature_to_index
    ]

    if missing_features:
        raise ValueError(
            f"Features required by {experiment} are absent from "
            f"the test data metadata: {missing_features}"
        )

    selected_indices = [
        feature_to_index[name]
        for name in saved_features
    ]

    # Select only the features used by this experiment.
    X_selected = X_test[:, :, selected_indices]

    if X_selected.shape[2] != int(checkpoint["input_size"]):
        raise ValueError(
            f"Selected {X_selected.shape[2]} features, but checkpoint "
            f"expects {checkpoint['input_size']}."
        )

    input_size, hidden_size, num_layers = infer_architecture(
        checkpoint
    )

    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.2,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # IMPORTANT:
    # y_test is already in original PM2.5/PM10 units.
    # Do not scale or inverse-transform these actual values.
    actual_values = y_test.copy()

    test_dataset = TensorDataset(
        torch.from_numpy(X_selected)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    scaled_predictions = []

    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(DEVICE)

            batch_predictions = model(batch_x)

            scaled_predictions.append(
                batch_predictions.cpu().numpy()
            )

    scaled_predictions = np.concatenate(
        scaled_predictions,
        axis=0,
    )

    # Inverse-transform predictions ONCE, using the statistics
    # saved with the corresponding checkpoint.
    target_mean = np.asarray(
        checkpoint["target_mean"],
        dtype=np.float64,
    ).reshape(1, 2)

    target_std = np.asarray(
        checkpoint["target_std"],
        dtype=np.float64,
    ).reshape(1, 2)

    if np.any(target_std <= 0):
        raise ValueError(
            f"Invalid target_std in {experiment} checkpoint: "
            f"{target_std}"
        )

    predicted_values = (
        scaled_predictions.astype(np.float64) * target_std
        + target_mean
    )

    if predicted_values.shape != actual_values.shape:
        raise ValueError(
            "Prediction/target shape mismatch: "
            f"{predicted_values.shape} vs {actual_values.shape}"
        )

    # Calculate metrics in the original pollutant units.
    target_names = ["PM2.5", "PM10"]

    experiment_results = {}

    for index, target_name in enumerate(target_names):
        actual = actual_values[:, index]
        predicted = predicted_values[:, index]

        mae = mean_absolute_error(actual, predicted)

        rmse = float(
            np.sqrt(mean_squared_error(actual, predicted))
        )

        r2 = r2_score(actual, predicted)

        experiment_results[target_name] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
        }

        print(f"\n{target_name}")
        print(f"  MAE : {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²  : {r2:.4f}")

    # Save columns in this order:
    # actual_PM2.5, actual_PM10, predicted_PM2.5, predicted_PM10
    combined_predictions = np.column_stack(
        [
            actual_values,
            predicted_values,
        ]
    )

    prediction_path = (
        ABLATION_DIR
        / f"{experiment}_test_predictions_corrected.npy"
    )

    np.save(prediction_path, combined_predictions)

    print(f"\nSaved predictions: {prediction_path}")

    return experiment_results


# ============================================================
# 6. COMPARE THE TWO EXPERIMENTS
# ============================================================

def print_comparison(results):
    experiments = ["pollution_only", "pollution_weather"]

    print("\n\n" + "=" * 90)
    print("CORRECTED ABLATION TEST COMPARISON")
    print("=" * 90)

    for pollutant in ["PM2.5", "PM10"]:
        print(f"\n{pollutant}")
        print(
            f"{'Metric':<10}"
            f"{'Pollution only':>20}"
            f"{'Pollution + weather':>24}"
        )
        print("-" * 54)

        for metric in ["MAE", "RMSE", "R2"]:
            pollution_only = results[
                experiments[0]
            ][pollutant][metric]

            pollution_weather = results[
                experiments[1]
            ][pollutant][metric]

            print(
                f"{metric:<10}"
                f"{pollution_only:>20.4f}"
                f"{pollution_weather:>24.4f}"
            )

        # Positive improvement means weather improved the metric.
        # For MAE/RMSE, lower is better.
        mae_without_weather = results[
            "pollution_only"
        ][pollutant]["MAE"]

        mae_with_weather = results[
            "pollution_weather"
        ][pollutant]["MAE"]

        improvement = mae_without_weather - mae_with_weather

        percentage = (
            improvement / mae_without_weather * 100
            if mae_without_weather != 0
            else float("nan")
        )

        print(
            f"\nMAE improvement from adding weather: "
            f"{improvement:.4f}"
        )
        print(
            f"Percentage MAE improvement: {percentage:.2f}%"
        )


# ============================================================
# 7. MAIN
# ============================================================

def main():
    print("Re-evaluating saved ablation checkpoints")
    print(f"Device: {DEVICE}")
    print(f"Project directory: {ROOT_DIR}")

    ABLATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_test, y_test, feature_names = load_test_data()

    results = {}

    for experiment in [
        "pollution_only",
        "pollution_weather",
    ]:
        results[experiment] = evaluate_checkpoint(
            experiment=experiment,
            X_test=X_test,
            y_test=y_test,
            all_feature_names=feature_names,
        )

    print_comparison(results)

    results_path = (
        ABLATION_DIR / "ablation_comparison_corrected.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Corrected metrics saved to: {results_path}")
    print(
        "Both saved checkpoints were evaluated without retraining."
    )


if __name__ == "__main__":
    main()
