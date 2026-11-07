"""
scripts/run_image_contribution_ablations.py

Evaluates the 5 image-contribution conditions on the fixed trained model:
1. Sensor-only fallback (mask=0)
2. Full hybrid with correctly matched images (mask=1)
3. Full hybrid with images masked (mask=0 with feats present)
4. Full hybrid with images shuffled across samples (break image-sensor pairing across 100 seeds)
5. Full hybrid with image features replaced by neutral representations:
   - Zero vector with mask=1
   - Mean training feature vector with mask=1

Saves detailed results and group breakdowns for the report.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
opt_dir = ROOT_DIR / "experiments/unified_hybrid/accuracy_optimization"

manifest_path = ROOT_DIR / "experiments/unified_hybrid/paired_dataset_hybrid.csv"
df = pd.read_csv(manifest_path)
train_df = df[df["split"] == "train"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)
N = len(test_df)

seqs = dict(np.load(ROOT_DIR / "experiments/multimodal_fusion/cached_sequences.npz"))
feats = dict(np.load(ROOT_DIR / "experiments/multimodal_fusion/cached_image_features.npz"))

# Compute training set mean feature vector for neutral ablation
train_feat_list = [feats[str(img_id)] for img_id in train_df["image_id"]]
mean_train_feat = np.mean(train_feat_list, axis=0, keepdims=True).astype(np.float32)

from experiments.unified_hybrid.accuracy_optimization.hybrid_model_optimized import OptimizedResidualHybridModel

model = OptimizedResidualHybridModel(
    lstm_checkpoint_path=str(ROOT_DIR / "models/lstm_baseline/best_lstm_model.pt"),
    freeze_sensor=True,
    visual_dim=32,
    dropout=0.2,
    max_delta_pm25=25.0,
    max_delta_pm10=40.0,
)
ckpt = torch.load(opt_dir / "checkpoints/best_optimized_hybrid_model.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

y_true_pm25 = test_df["station_target_pm25"].values
y_true_pm10 = test_df["station_target_pm10"].values

def eval_condition(get_feat_fn, get_mask_fn, desc=""):
    preds_25, preds_10, alphas, deltas_25, deltas_10 = [], [], [], [], []
    with torch.no_grad():
        for i, row in test_df.iterrows():
            g = row["station_date_group"]
            img_id = str(row["image_id"])
            x_seq = torch.from_numpy(seqs[g].astype(np.float32)).unsqueeze(0)
            x_feat = torch.from_numpy(get_feat_fn(i, img_id)).float()
            dist = torch.tensor([[float(row["distance_km"])]], dtype=torch.float32)
            mask = torch.tensor([[get_mask_fn(i)]], dtype=torch.float32)

            yp, a, det = model(x_seq, x_feat, dist, mask)
            preds_25.append(yp.numpy()[0, 0])
            preds_10.append(yp.numpy()[0, 1])
            alphas.append(a.numpy()[0, 0])
            deltas_25.append(det["delta"].numpy()[0, 0])
            deltas_10.append(det["delta"].numpy()[0, 1])

    preds_25 = np.array(preds_25)
    preds_10 = np.array(preds_10)
    alphas = np.array(alphas)
    deltas_25 = np.array(deltas_25)

    mae25 = float(np.mean(np.abs(y_true_pm25 - preds_25)))
    rmse25 = float(np.sqrt(np.mean((y_true_pm25 - preds_25)**2)))
    ss_tot25 = float(np.sum((y_true_pm25 - np.mean(y_true_pm25))**2))
    r2_25 = float(1.0 - np.sum((y_true_pm25 - preds_25)**2) / ss_tot25)

    mae10 = float(np.mean(np.abs(y_true_pm10 - preds_10)))
    rmse10 = float(np.sqrt(np.mean((y_true_pm10 - preds_10)**2)))
    ss_tot10 = float(np.sum((y_true_pm10 - np.mean(y_true_pm10))**2))
    r2_10 = float(1.0 - np.sum((y_true_pm10 - preds_10)**2) / ss_tot10)

    return {
        "description": desc,
        "pm25_mae": round(mae25, 4),
        "pm25_rmse": round(rmse25, 4),
        "pm25_r2": round(r2_25, 4),
        "pm10_mae": round(mae10, 4),
        "pm10_rmse": round(rmse10, 4),
        "pm10_r2": round(r2_10, 4),
        "mean_alpha": round(float(np.mean(alphas)), 4),
        "min_alpha": round(float(np.min(alphas)), 4),
        "max_alpha": round(float(np.max(alphas)), 4),
        "mean_delta_pm25": round(float(np.mean(deltas_25)), 4),
        "preds_pm25": preds_25,
        "alphas": alphas,
        "deltas_pm25": deltas_25,
    }

# 1. Condition A: Sensor-Only Fallback (mask=0)
res_fallback = eval_condition(
    lambda i, img_id: np.zeros((1, 2048), dtype=np.float32),
    lambda i: 0.0,
    desc="Sensor-Only Fallback (mask=0, zero vector)",
)

# 2. Condition B: Full Hybrid with Matched Images
res_matched = eval_condition(
    lambda i, img_id: feats[img_id].reshape(1, 2048).astype(np.float32),
    lambda i: 1.0,
    desc="Full Hybrid with Matched Images (mask=1)",
)

# 3. Condition C: Full Hybrid with Images Masked
res_masked = eval_condition(
    lambda i, img_id: feats[img_id].reshape(1, 2048).astype(np.float32),
    lambda i: 0.0,
    desc="Full Hybrid with Images Masked (mask=0, actual features present)",
)

# 4. Condition D: Full Hybrid with Images Shuffled (Monte Carlo 50 seeds)
test_img_ids = [str(r["image_id"]) for _, r in test_df.iterrows()]
shuffled_maes_25 = []
shuffled_maes_10 = []
shuffled_alphas = []

rng = np.random.RandomState(42)
for s in range(50):
    perm = rng.permutation(N)
    res_shuff = eval_condition(
        lambda i, img_id: feats[test_img_ids[perm[i]]].reshape(1, 2048).astype(np.float32),
        lambda i: 1.0,
        desc=f"Shuffled seed {s}",
    )
    shuffled_maes_25.append(res_shuff["pm25_mae"])
    shuffled_maes_10.append(res_shuff["pm10_mae"])
    shuffled_alphas.append(res_shuff["mean_alpha"])

res_shuffled_summary = {
    "description": "Full Hybrid with Images Shuffled across Test Instances (50 permutations)",
    "pm25_mae_mean": round(float(np.mean(shuffled_maes_25)), 4),
    "pm25_mae_std": round(float(np.std(shuffled_maes_25)), 4),
    "pm25_mae_min": round(float(np.min(shuffled_maes_25)), 4),
    "pm25_mae_max": round(float(np.max(shuffled_maes_25)), 4),
    "pm10_mae_mean": round(float(np.mean(shuffled_maes_10)), 4),
    "mean_alpha": round(float(np.mean(shuffled_alphas)), 4),
}

# 5. Condition E1: Neutral Zero Representation with mask=1
res_neutral_zero = eval_condition(
    lambda i, img_id: np.zeros((1, 2048), dtype=np.float32),
    lambda i: 1.0,
    desc="Neutral Zero Representation (zero vector with mask=1)",
)

# 6. Condition E2: Neutral Mean Representation with mask=1
res_neutral_mean = eval_condition(
    lambda i, img_id: mean_train_feat,
    lambda i: 1.0,
    desc="Neutral Mean Representation (training set centroid with mask=1)",
)

ablations_output = {
    "test_cohort_size": N,
    "conditions": {
        "condition_A_sensor_fallback": {k: v for k, v in res_fallback.items() if not isinstance(v, np.ndarray)},
        "condition_B_matched_images": {k: v for k, v in res_matched.items() if not isinstance(v, np.ndarray)},
        "condition_C_masked_images": {k: v for k, v in res_masked.items() if not isinstance(v, np.ndarray)},
        "condition_D_shuffled_images": res_shuffled_summary,
        "condition_E1_neutral_zero": {k: v for k, v in res_neutral_zero.items() if not isinstance(v, np.ndarray)},
        "condition_E2_neutral_mean": {k: v for k, v in res_neutral_mean.items() if not isinstance(v, np.ndarray)},
    }
}

with open(opt_dir / "image_branch_ablations.json", "w") as f:
    json.dump(ablations_output, f, indent=2)

print("Image branch ablations completed successfully:")
print(json.dumps(ablations_output, indent=2))
