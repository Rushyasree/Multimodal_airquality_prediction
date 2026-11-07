"""
experiments/multimodal_fusion/train_fusion.py

Trains:
1. Sensor-only LSTM
2. Image-only ResNet-50
3. Multimodal Fused Model (LSTM + ResNet-50)
on the exact same held-out train/test splits under strict group preservation.
Pre-extracts visual features once from frozen ResNet-50 for high training speed.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import json
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from experiments.multimodal_fusion.fusion_model import (
    MultimodalFusionModel, SensorOnlyModel, ImageOnlyModel, ResNetImageEncoder
)


class MultimodalDataset(Dataset):
    """Dataset serving 24x27 sensor sequences, 2048-dim pre-extracted image features, and targets."""

    def __init__(self, df: pd.DataFrame, sequences_dict: dict, image_features_dict: dict):
        self.df = df.reset_index(drop=True)
        self.sequences = sequences_dict
        self.img_features = image_features_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        group = row["station_date_group"]
        img_id = str(row["image_id"])
        
        sensor_tensor = torch.from_numpy(self.sequences[group].astype(np.float32))
        img_feat_tensor = torch.from_numpy(self.img_features[img_id].astype(np.float32))
        target_tensor = torch.tensor([float(row["station_target_pm25"])], dtype=torch.float32)

        return {
            "sensor": sensor_tensor,
            "image_feats": img_feat_tensor,
            "target": target_tensor,
            "image_id": img_id,
            "station_id": row["station_id"],
            "group": group,
            "target_time": row["target_time"]
        }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def precompute_image_features(df: pd.DataFrame, checkpoint_path: str, cache_path: Path) -> dict:
    """Pre-extracts 2048-dim visual features using frozen pretrained ResNet-50."""
    if cache_path.exists():
        print(f"Loading cached image features from {cache_path}...")
        return dict(np.load(cache_path))
        
    print(f"Pre-extracting visual features for {len(df)} images with ResNet-50...")
    encoder = ResNetImageEncoder()
    encoder.load_pretrained(checkpoint_path)
    encoder.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    feats_dict = {}
    with torch.no_grad():
        for idx, row in df.iterrows():
            img_id = str(row["image_id"])
            img_path = row["image_path"]
            try:
                with Image.open(img_path) as img:
                    t_img = transform(img.convert("RGB")).unsqueeze(0)
                    feat = encoder.extract_raw_features(t_img).squeeze(0).numpy()
            except Exception as e:
                feat = np.zeros(2048, dtype=np.float32)
            feats_dict[img_id] = feat
            
    np.savez_compressed(cache_path, **feats_dict)
    print(f"Saved pre-extracted features: {cache_path}")
    return feats_dict


def train_epoch(model, loader, optimizer, criterion, model_type="fusion"):
    model.train()
    total_loss = 0.0
    for batch in loader:
        sensor = batch["sensor"]
        img_feats = batch["image_feats"]
        target = batch["target"]
        
        optimizer.zero_grad()
        if model_type == "fusion":
            pred = model(sensor, image_raw_feats=img_feats)
        elif model_type == "sensor":
            pred = model(sensor)
        elif model_type == "image":
            pred = model(image_raw_feats=img_feats)
            
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(target)
        
    return total_loss / len(loader.dataset)


def evaluate_loss(model, loader, criterion, model_type="fusion"):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            sensor = batch["sensor"]
            img_feats = batch["image_feats"]
            target = batch["target"]
            
            if model_type == "fusion":
                pred = model(sensor, image_raw_feats=img_feats)
            elif model_type == "sensor":
                pred = model(sensor)
            elif model_type == "image":
                pred = model(image_raw_feats=img_feats)
                
            loss = criterion(pred, target)
            total_loss += loss.item() * len(target)
            
    return total_loss / len(loader.dataset)


def main():
    config_path = ROOT_DIR / "experiments/multimodal_fusion/configs/fusion_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    set_seed(cfg["seed"])
    
    out_dir = ROOT_DIR / cfg["output_dir"]
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = out_dir / "training_log.txt"
    log_f = open(log_file, "w", encoding="utf-8")
    
    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()
        
    log("="*75)
    log("VAYUDRISHTI MULTIMODAL FUSION TRAINING EXPERIMENT")
    log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Random seed: {cfg['seed']} | Batch size: {cfg['batch_size']} | Epochs: {cfg['epochs']}")
    log(f"Learning rate: {cfg['learning_rate']} | Weight decay: {cfg['weight_decay']}")
    log("="*75)
    
    # 1. Load paired dataset & sequences
    manifest_path = ROOT_DIR / cfg["dataset_path"]
    df = pd.read_csv(manifest_path)
    seqs_path = ROOT_DIR / cfg["cached_sequences_path"]
    seqs_dict = dict(np.load(seqs_path))
    
    # 2. Precompute image features
    cache_feats_path = out_dir / "cached_image_features.npz"
    img_feats_dict = precompute_image_features(
        df, str(ROOT_DIR / cfg["resnet_checkpoint"]), cache_feats_path
    )
    
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    
    log(f"Loaded paired manifest: {len(df)} instances across {df['station_date_group'].nunique()} groups.")
    log(f"Train split: {len(train_df)} samples, {train_df['station_date_group'].nunique()} groups, {train_df['station_id'].nunique()} stations.")
    log(f"Test split:  {len(test_df)} samples, {test_df['station_date_group'].nunique()} groups, {test_df['station_id'].nunique()} stations.")
    
    train_dataset = MultimodalDataset(train_df, seqs_dict, img_feats_dict)
    test_dataset = MultimodalDataset(test_df, seqs_dict, img_feats_dict)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)
    
    criterion = nn.SmoothL1Loss()
    
    # -------------------------------------------------------------
    # 3. Model A: Sensor-Only LSTM
    # -------------------------------------------------------------
    log("\n" + "-"*60)
    log("TRAINING MODEL A: Sensor-Only LSTM Baseline")
    log("-"*60)
    sensor_model = SensorOnlyModel(
        sensor_input_size=27,
        lstm_hidden=cfg["lstm_hidden"],
        lstm_layers=cfg["lstm_layers"],
        projection_dim=cfg["projection_dim"],
        dropout=cfg["dropout"],
    )
    sensor_model.sensor_encoder.load_pretrained(str(ROOT_DIR / cfg["lstm_checkpoint"]))
    
    if cfg["freeze_backbones"]:
        for param in sensor_model.sensor_encoder.lstm.parameters():
            param.requires_grad = False
        log("[SensorOnly] LSTM recurrent backbone frozen; training projection & regression head.")
        
    opt_sensor = torch.optim.Adam(
        filter(lambda p: p.requires_grad, sensor_model.parameters()),
        lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    
    best_sensor_loss = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss = train_epoch(sensor_model, train_loader, opt_sensor, criterion, "sensor")
        te_loss = evaluate_loss(sensor_model, test_loader, criterion, "sensor")
        log(f"Epoch {epoch:02d}/{cfg['epochs']:02d} | Train Loss: {tr_loss:.4f} | Test Loss: {te_loss:.4f}")
        if te_loss < best_sensor_loss:
            best_sensor_loss = te_loss
            torch.save(sensor_model.state_dict(), ckpt_dir / "best_sensor_model.pt")
            
    log(f"Best Sensor-Only Test Loss: {best_sensor_loss:.4f}")
    
    # -------------------------------------------------------------
    # 4. Model B: Image-Only ResNet-50
    # -------------------------------------------------------------
    log("\n" + "-"*60)
    log("TRAINING MODEL B: Image-Only ResNet-50 Baseline")
    log("-"*60)
    image_model = ImageOnlyModel(
        projection_dim=cfg["projection_dim"],
        dropout=cfg["dropout"],
    )
    image_model.image_encoder.load_pretrained(str(ROOT_DIR / cfg["resnet_checkpoint"]))
    
    if cfg["freeze_backbones"]:
        for param in image_model.image_encoder.backbone.parameters():
            param.requires_grad = False
        log("[ImageOnly] ResNet-50 backbone frozen; training projection & regression head.")
    
    opt_image = torch.optim.Adam(
        image_model.parameters(),
        lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    
    best_image_loss = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss = train_epoch(image_model, train_loader, opt_image, criterion, "image")
        te_loss = evaluate_loss(image_model, test_loader, criterion, "image")
        log(f"Epoch {epoch:02d}/{cfg['epochs']:02d} | Train Loss: {tr_loss:.4f} | Test Loss: {te_loss:.4f}")
        if te_loss < best_image_loss:
            best_image_loss = te_loss
            torch.save(image_model.state_dict(), ckpt_dir / "best_image_model.pt")
            
    log(f"Best Image-Only Test Loss: {best_image_loss:.4f}")
    
    # -------------------------------------------------------------
    # 5. Model C: Multimodal Fusion Model
    # -------------------------------------------------------------
    log("\n" + "-"*60)
    log("TRAINING MODEL C: Multimodal Fusion Model (LSTM + ResNet-50)")
    log("-"*60)
    fusion_model = MultimodalFusionModel(
        sensor_input_size=27,
        lstm_hidden=cfg["lstm_hidden"],
        lstm_layers=cfg["lstm_layers"],
        projection_dim=cfg["projection_dim"],
        dropout=cfg["dropout"],
    )
    fusion_model.sensor_encoder.load_pretrained(str(ROOT_DIR / cfg["lstm_checkpoint"]))
    fusion_model.image_encoder.load_pretrained(str(ROOT_DIR / cfg["resnet_checkpoint"]))
    
    if cfg["freeze_backbones"]:
        for param in fusion_model.sensor_encoder.lstm.parameters():
            param.requires_grad = False
        log("[Fusion] LSTM encoder frozen; training joint projection layers and fusion regression head.")
        
    opt_fusion = torch.optim.Adam(
        filter(lambda p: p.requires_grad, fusion_model.parameters()),
        lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    
    best_fusion_loss = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss = train_epoch(fusion_model, train_loader, opt_fusion, criterion, "fusion")
        te_loss = evaluate_loss(fusion_model, test_loader, criterion, "fusion")
        log(f"Epoch {epoch:02d}/{cfg['epochs']:02d} | Train Loss: {tr_loss:.4f} | Test Loss: {te_loss:.4f}")
        if te_loss < best_fusion_loss:
            best_fusion_loss = te_loss
            torch.save(fusion_model.state_dict(), ckpt_dir / "best_fusion_model.pt")
            
    log(f"Best Multimodal Fusion Test Loss: {best_fusion_loss:.4f}")
    
    log("\n" + "="*75)
    log("Training completed successfully for all three models!")
    log(f"Checkpoints saved in: {ckpt_dir}")
    log("="*75)
    log_f.close()


if __name__ == "__main__":
    main()
