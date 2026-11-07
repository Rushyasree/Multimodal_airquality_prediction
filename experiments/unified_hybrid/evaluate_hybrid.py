"""
experiments/unified_hybrid/evaluate_hybrid.py

Stand-alone evaluation script for VayuDrishti Unified Hybrid Architecture.
Evaluates the saved validation-selected checkpoint:
  experiments/unified_hybrid/checkpoints/best_unified_hybrid_model.pt

On:
1. Held-out untouched Test Set (N = 97 samples across 9 groups)
2. Validation Set (N = 21 samples across 6 groups)
3. Training Set (N = 81 samples across 18 groups)

Evaluates:
- Mode 1: Full Multimodal (mask_visual = 1.0)
- Mode 2: Missing-Modality Fallback (mask_visual = 0.0)

Generates:
- predictions.csv
- metrics.json
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.unified_hybrid.hybrid_model import UnifiedHybridModel
from experiments.unified_hybrid.train_hybrid import HybridDataset, compute_metrics, evaluate_cohort


def main():
    exp_dir = ROOT_DIR / "experiments/unified_hybrid"
    ckpt_path = exp_dir / "checkpoints/best_unified_hybrid_model.pt"

    assert ckpt_path.exists(), f"Checkpoint not found at: {ckpt_path}"

    manifest_path = exp_dir / "paired_dataset_hybrid.csv"
    df = pd.read_csv(manifest_path)

    seqs_path = ROOT_DIR / "experiments/multimodal_fusion/cached_sequences.npz"
    seqs_dict = dict(np.load(seqs_path))
    cache_feats_path = ROOT_DIR / "experiments/multimodal_fusion/cached_image_features.npz"
    img_feats_dict = dict(np.load(cache_feats_path))

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    test_dataset = HybridDataset(test_df, seqs_dict, img_feats_dict, modality_dropout=0.0)
    val_dataset = HybridDataset(val_df, seqs_dict, img_feats_dict, modality_dropout=0.0)
    train_dataset = HybridDataset(train_df, seqs_dict, img_feats_dict, modality_dropout=0.0)

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)

    model = UnifiedHybridModel(
        sensor_input_size=27,
        lstm_hidden=64,
        lstm_layers=2,
        projection_dim=64,
        dropout=0.2,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    # 1. Evaluate on Test set (N = 97)
    res_test_multi = evaluate_cohort(test_loader, model, force_mask=1.0)
    res_test_fall = evaluate_cohort(test_loader, model, force_mask=0.0)

    # 2. Evaluate on Validation set (N = 21)
    res_val_multi = evaluate_cohort(val_loader, model, force_mask=1.0)
    res_val_fall = evaluate_cohort(val_loader, model, force_mask=0.0)

    # 3. Evaluate on Training set (N = 81)
    res_train_multi = evaluate_cohort(train_loader, model, force_mask=1.0)
    res_train_fall = evaluate_cohort(train_loader, model, force_mask=0.0)

    print("=" * 80)
    print("UNIFIED HYBRID STANDALONE EVALUATION RESULTS")
    print("=" * 80)
    print(f"\n--- HELD-OUT TEST COHORT (N = {len(test_df)}, {test_df['station_date_group'].nunique()} groups, untouched) ---")
    print("MODE 1: Full Multimodal (Sensor + Image Gated):")
    print(f"  PM2.5: MAE = {res_test_multi['metrics_pm25']['MAE']:.2f} ug/m3 | RMSE = {res_test_multi['metrics_pm25']['RMSE']:.2f} | R2 = {res_test_multi['metrics_pm25']['R2']:.4f}")
    print(f"  PM10:  MAE = {res_test_multi['metrics_pm10']['MAE']:.2f} ug/m3 | RMSE = {res_test_multi['metrics_pm10']['RMSE']:.2f} | R2 = {res_test_multi['metrics_pm10']['R2']:.4f}")
    print(f"  Mean Gating Alpha: {float(np.mean(res_test_multi['alphas'])):.4f}")

    print("\nMODE 2: Fallback Mode (Sensor Telemetry Only, Image Masked = 0):")
    print(f"  PM2.5: MAE = {res_test_fall['metrics_pm25']['MAE']:.2f} ug/m3 | RMSE = {res_test_fall['metrics_pm25']['RMSE']:.2f} | R2 = {res_test_fall['metrics_pm25']['R2']:.4f}")
    print(f"  PM10:  MAE = {res_test_fall['metrics_pm10']['MAE']:.2f} ug/m3 | RMSE = {res_test_fall['metrics_pm10']['RMSE']:.2f} | R2 = {res_test_fall['metrics_pm10']['R2']:.4f}")
    print(f"  Mean Gating Alpha: {float(np.mean(res_test_fall['alphas'])):.4f}")

    print(f"\n--- VALIDATION COHORT (N = {len(val_df)}, {val_df['station_date_group'].nunique()} groups) ---")
    print(f"Full Multimodal PM2.5 MAE: {res_val_multi['metrics_pm25']['MAE']:.2f} | Fallback PM2.5 MAE: {res_val_fall['metrics_pm25']['MAE']:.2f}")

    # Save test predictions CSV
    test_df["ground_truth_pm25"] = res_test_multi["y_true"][:, 0]
    test_df["ground_truth_pm10"] = res_test_multi["y_true"][:, 1]
    test_df["pred_hybrid_pm25"] = res_test_multi["preds"][:, 0]
    test_df["pred_hybrid_pm10"] = res_test_multi["preds"][:, 1]
    test_df["pred_fallback_pm25"] = res_test_fall["preds"][:, 0]
    test_df["pred_fallback_pm10"] = res_test_fall["preds"][:, 1]
    test_df["gating_alpha"] = res_test_multi["alphas"][:, 0]

    save_cols = [
        "image_id", "station_id", "station_date_group", "capture_date", "target_time",
        "ground_truth_pm25", "ground_truth_pm10",
        "pred_hybrid_pm25", "pred_hybrid_pm10",
        "pred_fallback_pm25", "pred_fallback_pm10",
        "gating_alpha"
    ]
    preds_path = exp_dir / "predictions.csv"
    test_df[save_cols].to_csv(preds_path, index=False)
    print(f"\nSaved test predictions to: {preds_path}")

    # Save metrics JSON
    metrics_data = {
        "model_version": "VayuDrishti Unified Hybrid v1.0",
        "checkpoint_path": str(ckpt_path.relative_to(ROOT_DIR)),
        "test_cohort": {
            "total_samples": len(test_df),
            "groups_count": int(test_df["station_date_group"].nunique()),
            "stations_count": int(test_df["station_id"].nunique()),
            "pm25_ground_truth": {
                "mean": float(np.mean(res_test_multi["y_true"][:, 0])),
                "std": float(np.std(res_test_multi["y_true"][:, 0])),
            },
            "pm10_ground_truth": {
                "mean": float(np.mean(res_test_multi["y_true"][:, 1])),
                "std": float(np.std(res_test_multi["y_true"][:, 1])),
            }
        },
        "validation_cohort": {
            "total_samples": len(val_df),
            "groups_count": int(val_df["station_date_group"].nunique()),
            "stations_count": int(val_df["station_id"].nunique()),
        },
        "training_cohort": {
            "total_samples": len(train_df),
            "groups_count": int(train_df["station_date_group"].nunique()),
            "stations_count": int(train_df["station_id"].nunique()),
        },
        "full_multimodal_metrics": {
            "pm25": res_test_multi["metrics_pm25"],
            "pm10": res_test_multi["metrics_pm10"],
            "mean_gating_alpha": float(np.mean(res_test_multi["alphas"])),
        },
        "fallback_sensor_only_metrics": {
            "pm25": res_test_fall["metrics_pm25"],
            "pm10": res_test_fall["metrics_pm10"],
            "mean_gating_alpha": float(np.mean(res_test_fall["alphas"])),
        },
        "validation_metrics": {
            "full_multimodal": {
                "pm25": res_val_multi["metrics_pm25"],
                "pm10": res_val_multi["metrics_pm10"],
            },
            "fallback": {
                "pm25": res_val_fall["metrics_pm25"],
                "pm10": res_val_fall["metrics_pm10"],
            }
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
    print(f"Saved verified metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
