"""
experiments/multimodal_fusion/demo.py

Standalone demonstration script for the VayuDrishti Multimodal Fusion Prototype.
Accepts a 24-hour sensor sequence and a compatible image, validates inputs,
runs inference through both branches and the fusion head, and outputs PM2.5 in ug/m3.

Usage:
    python experiments/multimodal_fusion/demo.py
    python experiments/multimodal_fusion/demo.py --sample_idx 5
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms

from experiments.multimodal_fusion.fusion_model import (
    MultimodalFusionModel, SensorOnlyModel, ImageOnlyModel
)


def load_and_preprocess_image(image_path: str) -> torch.Tensor:
    """Validates and preprocesses an image for ResNet-50."""
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with Image.open(p) as img:
        img_rgb = img.convert("RGB")
        tensor = transform(img_rgb).unsqueeze(0)  # [1, 3, 224, 224]
        
    return tensor


def run_demo(sample_idx: int = 0):
    exp_dir = ROOT_DIR / "experiments/multimodal_fusion"
    ckpt_dir = exp_dir / "checkpoints"
    
    # 1. Load paired dataset
    dataset_path = exp_dir / "paired_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Paired dataset not found at {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    if sample_idx >= len(df) or sample_idx < 0:
        sample_idx = 0
        
    sample = df.iloc[sample_idx]
    group = sample["station_date_group"]
    img_path = sample["image_path"]
    
    # 2. Load cached 24h sequence
    seqs_path = exp_dir / "cached_sequences.npz"
    seqs_dict = np.load(seqs_path)
    if group not in seqs_dict:
        raise KeyError(f"Group {group} not in cached sequences!")
    sensor_arr = seqs_dict[group].astype(np.float32)  # [24, 27]
    sensor_tensor = torch.from_numpy(sensor_arr).unsqueeze(0)  # [1, 24, 27]
    
    # 3. Load Image
    image_tensor = load_and_preprocess_image(img_path)
    
    # 4. Load Models
    fusion_model = MultimodalFusionModel()
    fusion_model.load_state_dict(torch.load(ckpt_dir / "best_fusion_model.pt", map_location="cpu"))
    fusion_model.eval()
    
    sensor_model = SensorOnlyModel()
    sensor_model.load_state_dict(torch.load(ckpt_dir / "best_sensor_model.pt", map_location="cpu"))
    sensor_model.eval()
    
    image_model = ImageOnlyModel()
    image_model.load_state_dict(torch.load(ckpt_dir / "best_image_model.pt", map_location="cpu"))
    image_model.eval()
    
    # 5. Run Inference
    with torch.no_grad():
        pred_fused = float(fusion_model(sensor_tensor, image_tensor).item())
        pred_sensor = float(sensor_model(sensor_tensor).item())
        pred_image = float(image_model(image_tensor).item())
        
    y_true = float(sample["station_target_pm25"])
    
    # 6. Display Formatted Output
    print("\n" + "="*70)
    print("VAYUDRISHTI MULTIMODAL FUSION INFERENCE DEMONSTRATION")
    print("="*70)
    print(f"Sample Index:            {sample_idx} ({sample['split'].upper()} set)")
    print(f"Station:                 {sample['station_id']} ({sample['station_name']}, {sample['station_city']})")
    print(f"Distance to Monitor:     {sample['distance_km']:.2f} km")
    print(f"Historical 24h Window:   {sample['history_start']} to {sample['history_end']}")
    print(f"Forecast Target Time:    {sample['target_time']}")
    print(f"Image Source File:       {Path(img_path).name}")
    print("-"*70)
    print("INPUT VALIDATION:")
    print(f"  Sensor Input Tensor:   {list(sensor_tensor.shape)} (24 hourly steps, 27 standardized features)")
    print(f"  Image Input Tensor:    {list(image_tensor.shape)} (3 channels, 224x224 RGB ImageNet normalized)")
    print("-"*70)
    print("MODEL PREDICTIONS:")
    print(f"  [1] Sensor-Only LSTM:         {pred_sensor:8.2f} ug/m3")
    print(f"  [2] Image-Only ResNet-50:     {pred_image:8.2f} ug/m3")
    print(f"  [3] Fused Multimodal Model:   {pred_fused:8.2f} ug/m3")
    print(f"  >>> Ground Truth Observed:    {y_true:8.2f} ug/m3")
    print(f"  Prediction Error (|fused-y|): {abs(pred_fused - y_true):8.2f} ug/m3")
    print("="*70)
    print("SCIENTIFIC DISCLAIMER:")
    print("  This demonstration validates the computational functionality of the PyTorch")
    print("  multimodal fusion architecture. Because the underlying image manifest lacks sub-daily")
    print("  capture-hour timestamps, this output serves as a representative midday architecture")
    print("  demonstration rather than a validated next-hour causal forecast.")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VayuDrishti Multimodal Fusion Demo")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index of sample to demonstrate (0 to 198)")
    args = parser.parse_args()
    run_demo(args.sample_idx)
