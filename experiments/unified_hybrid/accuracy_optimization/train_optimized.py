"""
experiments/unified_hybrid/accuracy_optimization/train_optimized.py

Training script for Optimized Residual Quality-Aware Multimodal Fusion Model.
- Validates strictly on disjoint validation cohort (21 samples, 6 station-date groups).
- Monitors validation loss only (zero test loss leakage).
- Normalizes multi-target loss across PM2.5 and PM10.
- Regularizes residual delta to prevent overfitting on small training cohort.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from experiments.unified_hybrid.accuracy_optimization.hybrid_model_optimized import OptimizedResidualHybridModel


class PairedDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sequences_dict: dict,
        image_features_dict: dict,
        modality_dropout: float = 0.0,
    ):
        self.df = df.reset_index(drop=True)
        self.sequences = sequences_dict
        self.img_features = image_features_dict
        self.modality_dropout = modality_dropout

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        group = row["station_date_group"]
        img_id = str(row["image_id"])

        sensor_tensor = torch.from_numpy(self.sequences[group].astype(np.float32))
        img_feat_tensor = torch.from_numpy(self.img_features[img_id].astype(np.float32))
        dist_tensor = torch.tensor([float(row["distance_km"])], dtype=torch.float32)

        # Modality dropout during training
        mask = 1.0
        if self.modality_dropout > 0.0 and np.random.rand() < self.modality_dropout:
            mask = 0.0

        mask_tensor = torch.tensor([mask], dtype=torch.float32)
        target_tensor = torch.tensor(
            [float(row["station_target_pm25"]), float(row["station_target_pm10"])],
            dtype=torch.float32,
        )

        return {
            "sensor": sensor_tensor,
            "image_feats": img_feat_tensor,
            "distance": dist_tensor,
            "mask_visual": mask_tensor,
            "target": target_tensor,
            "group": group,
            "station_id": row["station_id"],
            "image_id": img_id,
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-8 else 0.0
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4), "unit": "ug/m3"}


def evaluate_loader(model, loader, force_mask=None):
    model.eval()
    targets_all = []
    preds_all = []
    sensor_all = []
    alphas_all = []
    deltas_all = []

    with torch.no_grad():
        for batch in loader:
            sensor = batch["sensor"]
            feats = batch["image_feats"]
            dist = batch["distance"]
            mask = batch["mask_visual"]
            if force_mask is not None:
                mask = torch.full_like(mask, force_mask)

            target = batch["target"]
            y_pred, alpha, details = model(sensor, feats, dist, mask)

            targets_all.append(target.numpy())
            preds_all.append(y_pred.numpy())
            sensor_all.append(details["y_sensor"].numpy())
            alphas_all.append(alpha.numpy())
            deltas_all.append(details["delta"].numpy())

    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    sensor_all = np.concatenate(sensor_all, axis=0)
    alphas_all = np.concatenate(alphas_all, axis=0)
    deltas_all = np.concatenate(deltas_all, axis=0)

    m_pm25 = compute_metrics(targets_all[:, 0], preds_all[:, 0])
    m_pm10 = compute_metrics(targets_all[:, 1], preds_all[:, 1])

    return {
        "targets": targets_all,
        "preds": preds_all,
        "sensor_preds": sensor_all,
        "alphas": alphas_all,
        "deltas": deltas_all,
        "metrics_pm25": m_pm25,
        "metrics_pm10": m_pm10,
    }


def train_model():
    opt_dir = ROOT_DIR / "experiments/unified_hybrid/accuracy_optimization"
    ckpt_dir = opt_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = ROOT_DIR / "experiments/unified_hybrid/paired_dataset_hybrid.csv"
    df = pd.read_csv(manifest_path)

    seqs = dict(np.load(ROOT_DIR / "experiments/multimodal_fusion/cached_sequences.npz"))
    feats = dict(np.load(ROOT_DIR / "experiments/multimodal_fusion/cached_image_features.npz"))

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    print(f"Paired Manifest: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Set seeds for deterministic training
    torch.manual_seed(42)
    np.random.seed(42)

    train_ds = PairedDataset(train_df, seqs, feats, modality_dropout=0.3)
    val_ds = PairedDataset(val_df, seqs, feats, modality_dropout=0.0)
    test_ds = PairedDataset(test_df, seqs, feats, modality_dropout=0.0)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    lstm_ckpt_path = str(ROOT_DIR / "models/lstm_baseline/best_lstm_model.pt")
    model = OptimizedResidualHybridModel(
        lstm_checkpoint_path=lstm_ckpt_path,
        freeze_sensor=True,
        visual_dim=32,
        dropout=0.2,
        max_delta_pm25=25.0,
        max_delta_pm10=40.0,
    )

    # Train only residual head and visual projector
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)

    criterion = nn.SmoothL1Loss()
    sigma_pm25 = 77.4174
    sigma_pm10 = 137.4705

    epochs = 25
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_path = ckpt_dir / "best_optimized_hybrid_model.pt"

    print("\n--- Training Optimized Residual Quality-Aware Model ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            sensor = batch["sensor"]
            img_feats = batch["image_feats"]
            dist = batch["distance"]
            mask = batch["mask_visual"]
            target = batch["target"]

            optimizer.zero_grad()
            preds, alpha, details = model(sensor, img_feats, dist, mask)

            # Normalized multi-target loss
            l_pm25 = criterion(preds[:, 0] / sigma_pm25, target[:, 0] / sigma_pm25)
            l_pm10 = criterion(preds[:, 1] / sigma_pm10, target[:, 1] / sigma_pm10)
            
            # L2 penalty on residual delta magnitude
            delta_reg = 0.01 * torch.mean(details["delta"] ** 2)
            
            loss = l_pm25 + l_pm10 + delta_reg
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(target)

        train_loss /= len(train_ds)
        scheduler.step()

        # Validation evaluation (strictly on val_ds)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sensor = batch["sensor"]
                img_feats = batch["image_feats"]
                dist = batch["distance"]
                mask = batch["mask_visual"]
                target = batch["target"]

                preds, alpha, details = model(sensor, img_feats, dist, mask)
                l_pm25 = criterion(preds[:, 0] / sigma_pm25, target[:, 0] / sigma_pm25)
                l_pm10 = criterion(preds[:, 1] / sigma_pm10, target[:, 1] / sigma_pm10)
                loss = l_pm25 + l_pm10
                val_loss += loss.item() * len(target)

        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "config": {
                        "visual_dim": 32,
                        "dropout": 0.2,
                        "max_delta_pm25": 25.0,
                        "max_delta_pm10": 40.0,
                    },
                },
                best_model_path,
            )
            saved_tag = "[SAVED BEST CKPT]"
        else:
            saved_tag = ""

        if epoch % 5 == 0 or epoch == 1 or saved_tag:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} {saved_tag}")

    print(f"\nTraining completed. Best model saved at Epoch {best_epoch} with Val Loss = {best_val_loss:.4f}")

    # Load best checkpoint for validation report
    best_ckpt = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(best_ckpt["model_state_dict"])
    val_res = evaluate_loader(model, val_loader)
    print("\nValidation Performance (Best Checkpoint):")
    print(f"PM2.5: MAE={val_res['metrics_pm25']['MAE']}, RMSE={val_res['metrics_pm25']['RMSE']}, R2={val_res['metrics_pm25']['R2']}")
    print(f"PM10:  MAE={val_res['metrics_pm10']['MAE']}, RMSE={val_res['metrics_pm10']['RMSE']}, R2={val_res['metrics_pm10']['R2']}")

    # Save config
    config = {
        "model_architecture": "OptimizedResidualHybridModel",
        "lstm_checkpoint": "models/lstm_baseline/best_lstm_model.pt",
        "freeze_sensor": True,
        "visual_dim": 32,
        "max_delta_pm25": 25.0,
        "max_delta_pm10": 40.0,
        "modality_dropout": 0.3,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "val_metrics": {
            "pm25": val_res["metrics_pm25"],
            "pm10": val_res["metrics_pm10"],
        },
    }
    with open(opt_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    train_model()
