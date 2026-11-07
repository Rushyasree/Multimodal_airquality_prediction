"""
experiments/multimodal_fusion/validation_v2/smoke_test.py

Verifies that the saved PyTorch checkpoints in validation_v2/checkpoints/
reproduce the exact predictions in validation_v2/predictions.csv without retraining.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch

from experiments.multimodal_fusion.fusion_model import (
    MultimodalFusionModel, SensorOnlyModel, ImageOnlyModel
)

def run_smoke_test():
    exp_dir = ROOT_DIR / "experiments/multimodal_fusion/validation_v2"
    ckpt_dir = exp_dir / "checkpoints"
    preds_df = pd.read_csv(exp_dir / "predictions.csv")
    seqs_dict = dict(np.load(ROOT_DIR / "experiments/multimodal_fusion/cached_sequences.npz"))
    img_feats_dict = dict(np.load(ROOT_DIR / "experiments/multimodal_fusion/cached_image_features.npz"))

    print("Running Smoke Test: Loading Checkpoints from validation_v2/checkpoints/...")

    sensor_m = SensorOnlyModel(27, 64, 2, 64, 0.2)
    sensor_m.load_state_dict(torch.load(ckpt_dir / "best_sensor_model.pt", map_location="cpu"))
    sensor_m.eval()

    img_m = ImageOnlyModel(64, 0.3)
    img_m.load_state_dict(torch.load(ckpt_dir / "best_image_model.pt", map_location="cpu"))
    img_m.eval()

    fus_m = MultimodalFusionModel(27, 64, 2, 64, 0.2)
    fus_m.load_state_dict(torch.load(ckpt_dir / "best_fusion_model.pt", map_location="cpu"))
    fus_m.eval()

    # Load all 97 test samples
    s_in = torch.stack([torch.from_numpy(seqs_dict[row["station_date_group"]].astype(np.float32)) for _, row in preds_df.iterrows()])
    i_in = torch.stack([torch.from_numpy(img_feats_dict[str(row["image_id"])].astype(np.float32)) for _, row in preds_df.iterrows()])

    with torch.no_grad():
        p_s = sensor_m(s_in).squeeze(-1).numpy()
        p_i = img_m(image_raw_feats=i_in).squeeze(-1).numpy()
        p_f = fus_m(s_in, image_raw_feats=i_in).squeeze(-1).numpy()

    diff_s = np.max(np.abs(p_s - preds_df["pred_sensor_only"].values))
    diff_i = np.max(np.abs(p_i - preds_df["pred_image_only"].values))
    diff_f = np.max(np.abs(p_f - preds_df["pred_fused"].values))

    print(f"Max abs diff Sensor-Only:  {diff_s:.8e} ug/m3")
    print(f"Max abs diff Image-Only:   {diff_i:.8e} ug/m3")
    print(f"Max abs diff Fusion Model: {diff_f:.8e} ug/m3")

    tolerance = 1e-4
    assert diff_s < tolerance, f"Sensor-only mismatch: {diff_s}"
    assert diff_i < tolerance, f"Image-only mismatch: {diff_i}"
    assert diff_f < tolerance, f"Fusion mismatch: {diff_f}"

    print("ALL 97 TEST ROWS REPRODUCED PERFECTLY! SMOKE TEST PASSED.")
    return True

if __name__ == "__main__":
    run_smoke_test()
