"""
experiments/unified_hybrid/train_hybrid.py

Trains and evaluates the VayuDrishti Unified Hybrid Architecture:
- Inputs: 24h x 27 sensor sequence + ResNet-50 visual features + missingness mask
- Targets: PM2.5 and PM10 simultaneously
- Validation-only checkpoint selection (zero test loss leakage)
- Evaluates both full multimodal and missing-modality fallback modes on held-out test cohort
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from experiments.unified_hybrid.hybrid_model import UnifiedHybridModel


class HybridDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sequences_dict: dict, image_features_dict: dict, modality_dropout: float = 0.0):
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
            "mask_visual": mask_tensor,
            "target": target_tensor,
            "image_id": img_id,
            "station_id": row["station_id"],
            "group": group,
            "target_time": row["target_time"],
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-8 else 0.0
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "unit": "ug/m3"}


def evaluate_cohort(loader, model, force_mask=None):
    model.eval()
    y_true_list, preds_list, alphas_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            sensor = batch["sensor"]
            img_feats = batch["image_feats"]
            mask = batch["mask_visual"]
            if force_mask is not None:
                mask = torch.full_like(mask, force_mask)

            target = batch["target"]
            preds, alpha = model(sensor, image_raw_feats=img_feats, mask_visual=mask)

            y_true_list.extend(target.numpy())
            preds_list.extend(preds.numpy())
            alphas_list.extend(alpha.numpy())

    y_true_arr = np.array(y_true_list)
    preds_arr = np.array(preds_list)

    metrics_pm25 = compute_metrics(y_true_arr[:, 0], preds_arr[:, 0])
    metrics_pm10 = compute_metrics(y_true_arr[:, 1], preds_arr[:, 1])

    return {
        "y_true": y_true_arr,
        "preds": preds_arr,
        "alphas": np.array(alphas_list),
        "metrics_pm25": metrics_pm25,
        "metrics_pm10": metrics_pm10,
    }


def main():
    exp_dir = ROOT_DIR / "experiments/unified_hybrid"
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = exp_dir / "paired_dataset_hybrid.csv"
    df = pd.read_csv(manifest_path)

    seqs_path = ROOT_DIR / "experiments/multimodal_fusion/cached_sequences.npz"
    seqs_dict = dict(np.load(seqs_path))
    cache_feats_path = ROOT_DIR / "experiments/multimodal_fusion/cached_image_features.npz"
    img_feats_dict = dict(np.load(cache_feats_path))

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    print(f"Loaded paired manifest: {len(df)} total instances.")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    train_dataset = HybridDataset(train_df, seqs_dict, img_feats_dict, modality_dropout=0.25)
    val_dataset = HybridDataset(val_df, seqs_dict, img_feats_dict, modality_dropout=0.0)
    test_dataset = HybridDataset(test_df, seqs_dict, img_feats_dict, modality_dropout=0.0)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = UnifiedHybridModel(
        sensor_input_size=27,
        lstm_hidden=64,
        lstm_layers=2,
        projection_dim=64,
        dropout=0.2,
    )
    model.sensor_encoder.load_pretrained(str(ROOT_DIR / "models/lstm_baseline/best_lstm_model.pt"))
    model.image_encoder.load_pretrained(str(ROOT_DIR / "models/pm25vision_cnn/best_model.pt"))

    # Freeze backbones
    for p in model.sensor_encoder.lstm.parameters():
        p.requires_grad = False
    for p in model.image_encoder.backbone.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0005,
        weight_decay=0.0001,
    )
    criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    best_epoch = -1
    epochs = 15

    print("\n--- Training Unified Hybrid Model ---")
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            sensor = batch["sensor"]
            img_feats = batch["image_feats"]
            mask = batch["mask_visual"]
            target = batch["target"]

            optimizer.zero_grad()
            preds, _ = model(sensor, image_raw_feats=img_feats, mask_visual=mask)
            # Weighted multi-target loss: PM2.5 + 0.5 * PM10
            l_pm25 = criterion(preds[:, 0], target[:, 0])
            l_pm10 = criterion(preds[:, 1], target[:, 1])
            loss = l_pm25 + 0.5 * l_pm10
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(target)

        tr_loss /= len(train_loader.dataset)

        # Evaluate validation loss
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sensor = batch["sensor"]
                img_feats = batch["image_feats"]
                mask = batch["mask_visual"]
                target = batch["target"]
                preds, _ = model(sensor, image_raw_feats=img_feats, mask_visual=mask)
                l_pm25 = criterion(preds[:, 0], target[:, 0])
                l_pm10 = criterion(preds[:, 1], target[:, 1])
                va_loss += (l_pm25 + 0.5 * l_pm10).item() * len(target)
        va_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best_unified_hybrid_model.pt")

    print(f"\nSelected Best Unified Hybrid Model at Epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")

    # Load best checkpoint for final evaluation
    model.load_state_dict(torch.load(ckpt_dir / "best_unified_hybrid_model.pt", map_location="cpu"))
    model.eval()

    # 1. Full Multimodal evaluation on test set (mask=1.0)
    res_multimodal = evaluate_cohort(test_loader, model, force_mask=1.0)
    # 2. Fallback mode evaluation on test set (mask=0.0)
    res_fallback = evaluate_cohort(test_loader, model, force_mask=0.0)

    print("\n" + "=" * 80)
    print("UNIFIED HYBRID EVALUATION RESULTS ON HELD-OUT TEST COHORT (N = 97)")
    print("=" * 80)
    print(f"MODE 1: Full Multimodal (Sensor + Image Gated):")
    print(f"  PM2.5: MAE = {res_multimodal['metrics_pm25']['MAE']:.2f} ug/m3 | RMSE = {res_multimodal['metrics_pm25']['RMSE']:.2f} | R2 = {res_multimodal['metrics_pm25']['R2']:.4f}")
    print(f"  PM10:  MAE = {res_multimodal['metrics_pm10']['MAE']:.2f} ug/m3 | RMSE = {res_multimodal['metrics_pm10']['RMSE']:.2f} | R2 = {res_multimodal['metrics_pm10']['R2']:.4f}")
    print(f"  Mean Gating Alpha: {float(np.mean(res_multimodal['alphas'])):.4f}")

    print(f"\nMODE 2: Fallback Mode (Sensor Telemetry Only, Image Masked = 0):")
    print(f"  PM2.5: MAE = {res_fallback['metrics_pm25']['MAE']:.2f} ug/m3 | RMSE = {res_fallback['metrics_pm25']['RMSE']:.2f} | R2 = {res_fallback['metrics_pm25']['R2']:.4f}")
    print(f"  PM10:  MAE = {res_fallback['metrics_pm10']['MAE']:.2f} ug/m3 | RMSE = {res_fallback['metrics_pm10']['RMSE']:.2f} | R2 = {res_fallback['metrics_pm10']['R2']:.4f}")
    print(f"  Mean Gating Alpha: {float(np.mean(res_fallback['alphas'])):.4f}")
    print("=" * 80)

    # Save test predictions CSV
    test_df["ground_truth_pm25"] = res_multimodal["y_true"][:, 0]
    test_df["ground_truth_pm10"] = res_multimodal["y_true"][:, 1]
    test_df["pred_hybrid_pm25"] = res_multimodal["preds"][:, 0]
    test_df["pred_hybrid_pm10"] = res_multimodal["preds"][:, 1]
    test_df["pred_fallback_pm25"] = res_fallback["preds"][:, 0]
    test_df["pred_fallback_pm10"] = res_fallback["preds"][:, 1]
    test_df["gating_alpha"] = res_multimodal["alphas"][:, 0]

    save_cols = [
        "image_id", "station_id", "station_date_group", "capture_date", "target_time",
        "ground_truth_pm25", "ground_truth_pm10",
        "pred_hybrid_pm25", "pred_hybrid_pm10",
        "pred_fallback_pm25", "pred_fallback_pm10",
        "gating_alpha"
    ]
    preds_path = exp_dir / "predictions.csv"
    test_df[save_cols].to_csv(preds_path, index=False)
    print(f"Saved: {preds_path}")

    # Save metrics JSON
    metrics_data = {
        "model_version": "VayuDrishti Unified Hybrid v1.0",
        "checkpoint_selected_epoch": best_epoch,
        "selection_validation_loss": best_val_loss,
        "test_cohort": {
            "total_samples": len(test_df),
            "groups_count": int(test_df["station_date_group"].nunique()),
            "stations_count": int(test_df["station_id"].nunique()),
            "pm25_ground_truth": {
                "mean": float(np.mean(res_multimodal["y_true"][:, 0])),
                "std": float(np.std(res_multimodal["y_true"][:, 0])),
            },
            "pm10_ground_truth": {
                "mean": float(np.mean(res_multimodal["y_true"][:, 1])),
                "std": float(np.std(res_multimodal["y_true"][:, 1])),
            }
        },
        "full_multimodal_metrics": {
            "pm25": res_multimodal["metrics_pm25"],
            "pm10": res_multimodal["metrics_pm10"],
            "mean_gating_alpha": float(np.mean(res_multimodal["alphas"])),
        },
        "fallback_sensor_only_metrics": {
            "pm25": res_fallback["metrics_pm25"],
            "pm10": res_fallback["metrics_pm10"],
            "mean_gating_alpha": float(np.mean(res_fallback["alphas"])),
        },
        "provenance": {
            "sensor_encoder": "2-Layer LSTM (frozen weights from best_lstm_model.pt)",
            "image_encoder": "ResNet-50 (frozen weights from best_model.pt)",
            "gating_module": "ModalityQualityGating with explicit zero-mask fallback",
            "prediction_head": "Multi-Layer Perceptron (129 -> 64 -> 32 -> 2) with Softplus activation"
        }
    }

    metrics_path = exp_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
