"""
train_pm25vision_cnn.py

Trains and evaluates an image-only CNN regression baseline using transfer learning
with a pretrained ResNet-50 backbone to predict continuous PM2.5 concentrations.

Key features:
- Uses PyTorch transfer learning with ResNet-50 (ImageNet weights).
- Strictly fits target normalization (mean and std) on the training set only.
- Validates each epoch and selects the best model checkpoint based on validation MAE.
- Evaluates test set strictly after model selection.
- Computes and reports continuous regression metrics: MAE, RMSE, and R^2.
- Saves model checkpoint, predictions CSV, evaluation metrics JSON, and training history.
- Adapted for CPU execution with multi-threading support.
"""

from pathlib import Path
import argparse
import copy
import json
import math
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# ---------------------------------------------------------
# MODEL ARCHITECTURE
# ---------------------------------------------------------
class ResNet50Regressor(nn.Module):
    """ResNet-50 backbone with MLP regression head for continuous PM2.5 prediction."""
    def __init__(self, pretrained=True, dropout=0.3, freeze_backbone=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base_model = models.resnet50(weights=weights)

        # Truncate final FC layer to use as feature extractor
        modules = list(base_model.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.regressor = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def extract_features(self, x):
        feats = self.backbone(x)
        return torch.flatten(feats, 1)

    def forward(self, x):
        feats = self.extract_features(x)
        return self.regressor(feats)


# ---------------------------------------------------------
# DATASET DEFINITION
# ---------------------------------------------------------
class PM25ImageDataset(Dataset):
    """Loads images on the fly with consistent resizing and normalization."""
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.paths = self.df["image_path"].values
        self.labels = self.df["pm25_label"].values.astype(np.float32)
        self.image_ids = self.df["image_id_clean"].values

        # Optional metadata if available
        self.station_ids = (
            self.df["nearest_station_id"].values if "nearest_station_id" in self.df.columns else [""] * len(self.df)
        )
        self.dates = (
            self.df["capture_date_clean"].values if "capture_date_clean" in self.df.columns else [""] * len(self.df)
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = Path(self.paths[idx])
        with Image.open(path) as img:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx], self.image_ids[idx]


# ---------------------------------------------------------
# FEATURE CACHING FOR EFFICIENT CPU TRAINING
# ---------------------------------------------------------
def cache_features(feature_extractor, data_loader, device):
    """
    Precompute ResNet-50 feature representations for CPU training speed.
    Returns:
        features: torch.Tensor of shape (N, 2048)
        labels: torch.Tensor of shape (N,)
        image_ids: list of str
    """
    feature_extractor.eval()
    all_feats = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for imgs, labels, ids in data_loader:
            imgs = imgs.to(device)
            feats = feature_extractor(imgs)
            feats = torch.flatten(feats, 1).cpu()
            all_feats.append(feats)
            all_labels.append(labels)
            all_ids.extend(ids)

    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0), all_ids


class FeatureDataset(Dataset):
    """Fast in-memory dataset of precomputed 2048-dim ResNet-50 features."""
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# ---------------------------------------------------------
# METRIC EVALUATION
# ---------------------------------------------------------
def calculate_metrics(y_true, y_pred):
    """Calculates continuous regression metrics: MAE, RMSE, and R^2."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))

    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }


# ---------------------------------------------------------
# TRAINING FUNCTION
# ---------------------------------------------------------
def train_model(args):
    start_time = time.time()
    ROOT = Path(__file__).resolve().parent

    data_dir = ROOT / args.data_dir
    model_dir = ROOT / args.model_dir
    results_dir = ROOT / args.results_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device configuration
    if torch.cuda.is_available() and not args.force_cpu:
        device = torch.device("cuda")
        print(f"Using GPU device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        threads = torch.get_num_threads()
        print(f"Using CPU device with {threads} PyTorch threads")

    print("=" * 70)
    print("PM25VISION IMAGE-ONLY CNN REGRESSION BASELINE (RESNET-50)")
    print("=" * 70)

    # 1. Load data manifests
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    if not val_path.exists():
        val_path = data_dir / "validation.csv"
    test_path = data_dir / "test.csv"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError(f"Missing split CSVs in {data_dir}. Run prepare_pm25vision_cnn_dataset.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    if args.smoke_test:
        print("\n*** SMOKE TEST MODE ACTIVATED (quick pipeline check) ***")
        train_df = train_df.head(16)
        val_df = val_df.head(8)
        test_df = test_df.head(8)
        args.epochs = 2

    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. Strict target normalization fit on training set only
    target_mean = float(train_df["pm25_label"].mean())
    target_std = float(train_df["pm25_label"].std())
    print(f"Target PM2.5 normalization (fitted on Train only): mean = {target_mean:.2f}, std = {target_std:.2f}")

    # 3. Vision transforms
    # Images consistently resized to 224x224 during loading; original files untouched
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_raw_ds = PM25ImageDataset(train_df, transform=val_test_transform)
    val_raw_ds = PM25ImageDataset(val_df, transform=val_test_transform)
    test_raw_ds = PM25ImageDataset(test_df, transform=val_test_transform)

    batch_size = args.batch_size
    train_raw_loader = DataLoader(train_raw_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_raw_loader = DataLoader(val_raw_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_raw_loader = DataLoader(test_raw_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Initialize model
    print("\nLoading pretrained ResNet-50 backbone...")
    full_model = ResNet50Regressor(pretrained=True, dropout=args.dropout, freeze_backbone=True).to(device)

    # 5. Extract features for CPU efficiency
    print("Extracting frozen ResNet-50 features...")
    t0 = time.time()
    train_feats, train_labels, train_ids = cache_features(full_model.backbone, train_raw_loader, device)
    val_feats, val_labels, val_ids = cache_features(full_model.backbone, val_raw_loader, device)
    test_feats, test_labels, test_ids = cache_features(full_model.backbone, test_raw_loader, device)
    print(f"Feature extraction completed in {time.time() - t0:.1f}s")
    print(f"Extracted feature shapes -> Train: {train_feats.shape}, Val: {val_feats.shape}, Test: {test_feats.shape}")

    # Scale targets using train statistics
    train_targets_norm = (train_labels - target_mean) / target_std
    val_targets_norm = (val_labels - target_mean) / target_std
    test_targets_norm = (test_labels - target_mean) / target_std

    train_feat_ds = FeatureDataset(train_feats, train_targets_norm)
    val_feat_ds = FeatureDataset(val_feats, val_targets_norm)

    train_loader = DataLoader(train_feat_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_feat_ds, batch_size=batch_size, shuffle=False)

    # 6. Optimization setup
    criterion = nn.SmoothL1Loss()  # Huber loss
    optimizer = torch.optim.AdamW(
        full_model.regressor.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 7. Training loop with model checkpoint selection
    print("\n" + "=" * 70)
    print(f"STARTING REGRESSION HEAD TRAINING (Max Epochs: {args.epochs}, Patience: {args.patience})")
    print("=" * 70)

    best_val_mae = float("inf")
    best_val_loss = float("inf")
    best_regressor_state = None
    best_epoch = 0
    patience_counter = 0

    history = []

    for epoch in range(1, args.epochs + 1):
        full_model.regressor.train()
        train_loss_sum = 0.0
        train_count = 0

        for feats_b, targets_b in train_loader:
            feats_b = feats_b.to(device)
            targets_b = targets_b.to(device).unsqueeze(1)

            optimizer.zero_grad()
            preds_b = full_model.regressor(feats_b)
            loss = criterion(preds_b, targets_b)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(feats_b)
            train_count += len(feats_b)

        train_loss = train_loss_sum / train_count

        # Validation evaluation
        full_model.regressor.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_preds_list = []
        val_actuals_list = []

        with torch.no_grad():
            for feats_b, targets_b in val_loader:
                feats_b = feats_b.to(device)
                targets_b = targets_b.to(device).unsqueeze(1)

                preds_b = full_model.regressor(feats_b)
                loss = criterion(preds_b, targets_b)

                val_loss_sum += loss.item() * len(feats_b)
                val_count += len(feats_b)

                # Convert back to raw PM2.5 units
                pred_raw = preds_b.cpu().numpy().squeeze(1) * target_std + target_mean
                actual_raw = targets_b.cpu().numpy().squeeze(1) * target_std + target_mean
                val_preds_list.extend(pred_raw)
                val_actuals_list.extend(actual_raw)

        val_loss = val_loss_sum / val_count
        val_metrics = calculate_metrics(val_actuals_list, val_preds_list)
        val_mae = val_metrics["MAE"]

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_mae": val_mae,
            "val_rmse": val_metrics["RMSE"],
            "val_r2": val_metrics["R2"],
            "lr": current_lr,
        })

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            best_val_loss = val_loss
            best_epoch = epoch
            best_regressor_state = copy.deepcopy(full_model.regressor.state_dict())
            patience_counter = 0
            flag = " [* BEST CHECKPOINT]"
        else:
            patience_counter += 1
            flag = ""

        print(f"Epoch {epoch:2d}/{args.epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f} | Val RMSE: {val_metrics['RMSE']:.2f} | "
              f"Val R^2: {val_metrics['R2']:.4f} | LR: {current_lr:.6f}{flag}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {args.patience} epochs).")
            break

    # 8. Load best checkpoint for final evaluation
    if best_regressor_state is not None:
        full_model.regressor.load_state_dict(best_regressor_state)
        print(f"\nRestored best model from Epoch {best_epoch} (Val MAE: {best_val_mae:.2f})")
    else:
        print("\nWarning: No best model state found, using final state.")

    # 9. Evaluate Test Set strictly after model selection
    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION")
    print("=" * 70)

    full_model.regressor.eval()
    with torch.no_grad():
        test_feats = test_feats.to(device)
        test_preds_norm = full_model.regressor(test_feats).cpu().numpy().squeeze(1)
        test_preds_raw = test_preds_norm * target_std + target_mean
        test_actuals_raw = test_labels.numpy()

    test_metrics = calculate_metrics(test_actuals_raw, test_preds_raw)

    print("\nPM25Vision Image-Only Baseline Test Metrics:")
    print("-" * 45)
    print(f"  MAE:   {test_metrics['MAE']:.4f} ug/m3")
    print(f"  RMSE:  {test_metrics['RMSE']:.4f} ug/m3")
    print(f"  R^2:   {test_metrics['R2']:.4f}")
    print("-" * 45)

    # 10. Save all artifacts
    # Save model checkpoint
    checkpoint_path = model_dir / "best_model.pt"
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": full_model.state_dict(),
        "regressor_state_dict": full_model.regressor.state_dict(),
        "target_mean": target_mean,
        "target_std": target_std,
        "test_metrics": test_metrics,
        "val_metrics": {
            "MAE": best_val_mae,
            "RMSE": history[best_epoch - 1]["val_rmse"],
            "R2": history[best_epoch - 1]["val_r2"],
        },
        "config": vars(args),
    }, checkpoint_path)
    print(f"\nSaved model checkpoint -> {checkpoint_path}")

    # Save test predictions table
    test_preds_df = pd.DataFrame({
        "image_id": test_ids,
        "pm25_actual": np.round(test_actuals_raw, 2),
        "pm25_predicted": np.round(test_preds_raw, 2),
        "absolute_error": np.round(np.abs(test_preds_raw - test_actuals_raw), 2),
        "squared_error": np.round((test_preds_raw - test_actuals_raw) ** 2, 2),
    })
    preds_csv_path = results_dir / "test_predictions.csv"
    test_preds_df.to_csv(preds_csv_path, index=False)
    print(f"Saved test predictions table -> {preds_csv_path}")

    # Save training history
    history_csv_path = results_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(history_csv_path, index=False)
    print(f"Saved training history -> {history_csv_path}")

    # Save full metrics JSON
    full_results = {
        "model_name": "ResNet-50 Transfer Learning Regressor",
        "modality": "Image-only (PM25Vision)",
        "backbone": "ResNet-50 (Pretrained ImageNet)",
        "device": str(device),
        "best_epoch": best_epoch,
        "target_preprocessing": {
            "strategy": "StandardScaler fitted on training split only",
            "mean": round(target_mean, 4),
            "std": round(target_std, 4),
        },
        "splits": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
        },
        "validation_metrics": {
            "MAE": round(best_val_mae, 4),
            "RMSE": round(history[best_epoch - 1]["val_rmse"], 4),
            "R2": round(history[best_epoch - 1]["val_r2"], 4),
        },
        "test_metrics": test_metrics,
        "execution_time_seconds": round(time.time() - start_time, 2),
    }

    metrics_json_path = results_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"Saved metrics summary -> {metrics_json_path}")

    print("\nCNN baseline training and evaluation completed successfully.")
    return full_results


# ---------------------------------------------------------
# CLI INTERFACE
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PM25Vision CNN Regression Baseline")
    parser.add_argument("--data-dir", type=str, default="data/processed/pm25vision_cnn", help="Data directory")
    parser.add_argument("--model-dir", type=str, default="models/pm25vision_cnn", help="Model checkpoint directory")
    parser.add_argument("--results-dir", type=str, default="results/pm25vision_cnn", help="Results directory")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2-epoch smoke test on small subset")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU execution")

    args = parser.parse_args()
    train_model(args)
