
from pathlib import Path
import json
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/processed/model_sequences")
MODEL_DIR = Path("models/lstm_baseline")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 256
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
MAX_EPOCHS = 20
PATIENCE = 5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 70)
print("LSTM BASELINE: PM2.5 AND PM10 FORECASTING")
print("=" * 70)
print("Device:", DEVICE)

# ============================================================
# DATASET
# ============================================================

class SequenceDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

        if len(self.X) != len(self.y):
            raise ValueError("Input and target lengths differ.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Copy the individual sample from the read-only memory map.
        x = torch.from_numpy(
            np.array(self.X[idx], dtype=np.float32, copy=True)
        )
        y = torch.from_numpy(
            np.array(self.y[idx], dtype=np.float32, copy=True)
        )
        return x, y


train_ds = SequenceDataset(
    DATA_DIR / "X_train.npy",
    DATA_DIR / "y_train.npy",
)
val_ds = SequenceDataset(
    DATA_DIR / "X_validation.npy",
    DATA_DIR / "y_validation.npy",
)
test_ds = SequenceDataset(
    DATA_DIR / "X_test.npy",
    DATA_DIR / "y_test.npy",
)

print(f"Train sequences: {len(train_ds):,}")
print(f"Validation sequences: {len(val_ds):,}")
print(f"Test sequences: {len(test_ds):,}")
print("Input shape per sample:", train_ds.X.shape[1:])

# Fit target scaling using training targets only.
y_train_raw = np.asarray(train_ds.y, dtype=np.float64)

target_mean = y_train_raw.mean(axis=0)
target_std = y_train_raw.std(axis=0)
target_std[target_std < 1e-8] = 1.0

del y_train_raw

def scale_target(y):
    return (y - target_mean) / target_std


class ScaledTargetDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        y = (y - torch.tensor(target_mean, dtype=torch.float32)) / (
            torch.tensor(target_std, dtype=torch.float32)
        )
        return x, y


train_scaled = ScaledTargetDataset(train_ds)
val_scaled = ScaledTargetDataset(val_ds)

train_loader = DataLoader(
    train_scaled,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)

val_loader = DataLoader(
    val_scaled,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)

# ============================================================
# MODEL
# ============================================================

class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        output_size=2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.regressor(last_hidden)


input_size = train_ds.X.shape[2]

model = LSTMForecaster(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)

print("\nModel architecture:")
print(model)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================================
# VALIDATION
# ============================================================

def run_validation():
    model.eval()
    total_loss = 0.0
    total_count = 0

    predictions = []
    actuals = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            pred = model(xb)
            loss = criterion(pred, yb)

            batch_count = len(xb)
            total_loss += loss.item() * batch_count
            total_count += batch_count

            # Return values to original PM concentration units.
            pred_raw = (
                pred.cpu().numpy() * target_std + target_mean
            )
            actual_raw = (
                yb.cpu().numpy() * target_std + target_mean
            )

            predictions.append(pred_raw)
            actuals.append(actual_raw)

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    return total_loss / total_count, predictions, actuals


def calculate_metrics(actual, predicted):
    result = {}

    for j, name in enumerate(["PM2.5", "PM10"]):
        error = predicted[:, j] - actual[:, j]

        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error ** 2))

        ss_res = np.sum(error ** 2)
        ss_tot = np.sum((actual[:, j] - actual[:, j].mean()) ** 2)

        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

        result[name] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2) if r2 is not None else None,
        }

    return result


# ============================================================
# TRAINING WITH EARLY STOPPING
# ============================================================

best_val_loss = float("inf")
best_state = None
epochs_without_improvement = 0
history = []

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        pred = model(xb)
        loss = criterion(pred, yb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_count = len(xb)
        total_loss += loss.item() * batch_count
        total_count += batch_count

        if batch_idx % 250 == 0:
            print(
                f"Epoch {epoch} | Batch {batch_idx} "
                f"| Loss {loss.item():.5f}"
            )

    train_loss = total_loss / total_count
    val_loss, val_pred, val_actual = run_validation()
    val_metrics = calculate_metrics(val_actual, val_pred)

    scheduler.step(val_loss)

    history.append({
        "epoch": epoch,
        "train_mse_scaled": train_loss,
        "validation_mse_scaled": val_loss,
        "validation_PM25_MAE": val_metrics["PM2.5"]["MAE"],
        "validation_PM25_RMSE": val_metrics["PM2.5"]["RMSE"],
        "validation_PM25_R2": val_metrics["PM2.5"]["R2"],
        "validation_PM10_MAE": val_metrics["PM10"]["MAE"],
        "validation_PM10_RMSE": val_metrics["PM10"]["RMSE"],
        "validation_PM10_R2": val_metrics["PM10"]["R2"],
    })

    print(f"\nEpoch {epoch}/{MAX_EPOCHS}")
    print(f"Train MSE (scaled): {train_loss:.6f}")
    print(f"Validation MSE (scaled): {val_loss:.6f}")
    print(
        f"PM2.5: MAE={val_metrics['PM2.5']['MAE']:.3f}, "
        f"RMSE={val_metrics['PM2.5']['RMSE']:.3f}, "
        f"R2={val_metrics['PM2.5']['R2']:.4f}"
    )
    print(
        f"PM10:  MAE={val_metrics['PM10']['MAE']:.3f}, "
        f"RMSE={val_metrics['PM10']['RMSE']:.3f}, "
        f"R2={val_metrics['PM10']['R2']:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": best_state,
                "input_size": input_size,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "target_mean": target_mean.tolist(),
                "target_std": target_std.tolist(),
            },
            MODEL_DIR / "best_lstm_model.pt",
        )

        print("Best model checkpoint saved.")
    else:
        epochs_without_improvement += 1
        print(
            f"No validation improvement: "
            f"{epochs_without_improvement}/{PATIENCE}"
        )

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break

# ============================================================
# TEST ONLY AFTER MODEL SELECTION
# ============================================================

if best_state is None:
    raise RuntimeError("No valid model checkpoint was produced.")

model.load_state_dict(best_state)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)

model.eval()
test_predictions = []
test_actuals = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)

        pred = model(xb).cpu().numpy()
        pred = pred * target_std + target_mean

        test_predictions.append(pred)
        test_actuals.append(yb.numpy())

test_predictions = np.concatenate(test_predictions)
test_actuals = np.concatenate(test_actuals)

test_metrics = calculate_metrics(
    test_actuals,
    test_predictions,
)

print("\n" + "=" * 70)
print("FINAL TEST METRICS")
print("=" * 70)

for target, metrics in test_metrics.items():
    print(f"\n{target}")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}" if value is not None
              else f"  {metric_name}: unavailable")

pd.DataFrame(history).to_csv(
    MODEL_DIR / "training_history.csv",
    index=False,
)

with open(MODEL_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(test_metrics, f, indent=2)

np.save(
    MODEL_DIR / "test_predictions.npy",
    test_predictions.astype(np.float32),
)

print("\nSaved:")
print(MODEL_DIR / "best_lstm_model.pt")
print(MODEL_DIR / "training_history.csv")
print(MODEL_DIR / "test_metrics.json")
print(MODEL_DIR / "test_predictions.npy")
print("\nLSTM baseline training completed.")
