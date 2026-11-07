"""
experiments/multimodal_fusion/validation_v2/train_v2.py

Trains:
1. Sensor-Only LSTM Baseline
2. Image-Only ResNet-50 Baseline
3. Multimodal Fusion Model (LSTM + ResNet-50)

CORRECTED METHODOLOGY (Validation v2):
- Model selection and checkpoint saving strictly monitor VALIDATION loss (val split: 21 samples, 6 groups).
- Test loss is NEVER used for model selection, early stopping, or checkpoint saving.
- Preprocessing and feature scaling are strictly derived from training split statistics.
- All models evaluated on the untouched test split (97 samples, 9 groups).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from experiments.multimodal_fusion.fusion_model import (
    MultimodalFusionModel, SensorOnlyModel, ImageOnlyModel
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
            "target_time": row["target_time"],
        }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


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
    exp_dir = ROOT_DIR / "experiments/multimodal_fusion/validation_v2"
    config_path = exp_dir / "configs/fusion_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    set_seed(cfg["seed"])

    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = exp_dir / "training_log.txt"
    log_f = open(log_file, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log("=" * 80)
    log("VAYUDRISHTI MULTIMODAL FUSION TRAINING (VALIDATION V2 - STRICT VALIDATION SELECTION)")
    log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Seed: {cfg['seed']} | Batch Size: {cfg['batch_size']} | Epochs: {cfg['epochs']}")
    log(f"LR: {cfg['learning_rate']} | Weight Decay: {cfg['weight_decay']}")
    log("=" * 80)

    # 1. Load paired dataset & sequences
    df = pd.read_csv(ROOT_DIR / cfg["dataset_path"])
    seqs_path = ROOT_DIR / cfg["cached_sequences_path"]
    seqs_dict = dict(np.load(seqs_path))
    cache_feats_path = ROOT_DIR / cfg["cached_image_features_path"]
    img_feats_dict = dict(np.load(cache_feats_path))

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    log(f"Loaded paired manifest: {len(df)} instances across {df['station_date_group'].nunique()} groups.")
    log(f"Train split: {len(train_df)} samples, {train_df['station_date_group'].nunique()} groups, {train_df['station_id'].nunique()} stations.")
    log(f"Val split:   {len(val_df)} samples, {val_df['station_date_group'].nunique()} groups, {val_df['station_id'].nunique()} stations.")
    log(f"Test split:  {len(test_df)} samples, {test_df['station_date_group'].nunique()} groups, {test_df['station_id'].nunique()} stations.")

    # Confirm strict zero group overlap
    tr_grps = set(train_df["station_date_group"])
    va_grps = set(val_df["station_date_group"])
    te_grps = set(test_df["station_date_group"])
    assert len(tr_grps.intersection(va_grps)) == 0, "Train-Val group overlap!"
    assert len(tr_grps.intersection(te_grps)) == 0, "Train-Test group overlap!"
    assert len(va_grps.intersection(te_grps)) == 0, "Val-Test group overlap!"
    log("Zero group overlap confirmed between all three splits.")

    train_dataset = MultimodalDataset(train_df, seqs_dict, img_feats_dict)
    val_dataset = MultimodalDataset(val_df, seqs_dict, img_feats_dict)
    test_dataset = MultimodalDataset(test_df, seqs_dict, img_feats_dict)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)

    criterion = nn.SmoothL1Loss()

    training_summary = {}

    # -------------------------------------------------------------
    # Model A: Sensor-Only LSTM
    # -------------------------------------------------------------
    log("\n" + "-" * 70)
    log("TRAINING MODEL A: Sensor-Only LSTM Baseline")
    log("-" * 70)
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
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    best_val_loss_sensor = float("inf")
    best_epoch_sensor = -1
    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss = train_epoch(sensor_model, train_loader, opt_sensor, criterion, "sensor")
        va_loss = evaluate_loss(sensor_model, val_loader, criterion, "sensor")
        log(f"Epoch {epoch:02d}/{cfg['epochs']:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f}")

        # Model selection strictly on validation loss
        if va_loss < best_val_loss_sensor:
            best_val_loss_sensor = va_loss
            best_epoch_sensor = epoch
            torch.save(sensor_model.state_dict(), ckpt_dir / "best_sensor_model.pt")

    log(f"Selected Sensor-Only Checkpoint at Epoch {best_epoch_sensor:02d} with Val Loss: {best_val_loss_sensor:.4f}")
    training_summary["sensor_only"] = {
        "best_epoch": best_epoch_sensor,
        "best_val_loss": best_val_loss_sensor,
    }

    # -------------------------------------------------------------
    # Model B: Image-Only ResNet-50
    # -------------------------------------------------------------
    log("\n" + "-" * 70)
    log("TRAINING MODEL B: Image-Only ResNet-50 Baseline")
    log("-" * 70)
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
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    best_val_loss_image = float("inf")
    best_epoch_image = -1
    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss = train_epoch(image_model, train_loader, opt_image, criterion, "image")
        va_loss = evaluate_loss(image_model, val_loader, criterion, "image")
        log(f"Epoch {epoch:02d}/{cfg['epochs']:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f}")

        # Model selection strictly on validation loss
        if va_loss < best_val_loss_image:
            best_val_loss_image = va_loss
            best_epoch_image = epoch
            torch.save(image_model.state_dict(), ckpt_dir / "best_image_model.pt")

    log(f"Selected Image-Only Checkpoint at Epoch {best_epoch_image:02d} with Val Loss: {best_val_loss_image:.4f}")
    training_summary["image_only"] = {
        "best_epoch": best_epoch_image,
        "best_val_loss": best_val_loss_image,
    }

    # -------------------------------------------------------------
    # Model C: Multimodal Fusion Model
    # -------------------------------------------------------------
    log("\n" + "-" * 70)
    log("TRAINING MODEL C: Multimodal Fusion Model (LSTM + ResNet-50)")
    log("-" * 70)
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
        log("[Fusion] LSTM encoder frozen; training joint projection layers and regression head.")

    opt_fusion = torch.optim.Adam(
        filter(lambda p: p.requires_grad, fusion_model.parameters()),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    best_val_loss_fusion = float("inf")
    best_epoch_fusion = -1
    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss = train_epoch(fusion_model, train_loader, opt_fusion, criterion, "fusion")
        va_loss = evaluate_loss(fusion_model, val_loader, criterion, "fusion")
        log(f"Epoch {epoch:02d}/{cfg['epochs']:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f}")

        # Model selection strictly on validation loss
        if va_loss < best_val_loss_fusion:
            best_val_loss_fusion = va_loss
            best_epoch_fusion = epoch
            torch.save(fusion_model.state_dict(), ckpt_dir / "best_fusion_model.pt")

    log(f"Selected Fusion Checkpoint at Epoch {best_epoch_fusion:02d} with Val Loss: {best_val_loss_fusion:.4f}")
    training_summary["fusion"] = {
        "best_epoch": best_epoch_fusion,
        "best_val_loss": best_val_loss_fusion,
    }

    log("\n" + "=" * 80)
    log("TRAINING COMPLETE: All checkpoints saved based exclusively on VALIDATION loss.")
    log(f"Checkpoints directory: {ckpt_dir}")
    log("=" * 80)
    log_f.close()

    summary_path = exp_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)


if __name__ == "__main__":
    main()
