"""
experiments/unified_hybrid/hybrid_model.py

PyTorch implementation of the VayuDrishti Unified Hybrid Architecture:
1. LSTMSensorEncoder: Encodes 24h x 27 sensor/weather/spatial/temporal sequence.
2. ResNetImageEncoder: Encodes 224x224 RGB camera image into visual representation.
3. ModalityQualityGating: Learned adaptive gating with explicit missingness masking.
4. UnifiedPredictionHead: Joint regression head predicting both PM2.5 and PM10 (ug/m3).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models


class LSTMSensorEncoder(nn.Module):
    """Encodes 24-hour sensor/weather/context sequence (24x27) into a 64-dim embedding."""

    def __init__(
        self,
        input_size: int = 27,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        projection_dim: int = 64,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.projection_dim = projection_dim

    def load_pretrained(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt["model_state_dict"]
        lstm_sd = {k: v for k, v in sd.items() if k.startswith("lstm.")}
        self.load_state_dict(lstm_sd, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.projection(last_hidden)


class ResNetImageEncoder(nn.Module):
    """Encodes 224x224 RGB camera image into a 64-dim embedding using ResNet-50."""

    def __init__(
        self,
        projection_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        base_resnet = models.resnet50(weights=None)
        modules = list(base_resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.projection = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, projection_dim),
            nn.ReLU(),
        )
        self.projection_dim = projection_dim

    def load_pretrained(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt["model_state_dict"]
        backbone_sd = {k: v for k, v in sd.items() if k.startswith("backbone.")}
        self.load_state_dict(backbone_sd, strict=False)

    def extract_raw_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return torch.flatten(feats, 1)

    def project_raw_features(self, raw_feats: torch.Tensor) -> torch.Tensor:
        return self.projection(raw_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_feats = self.extract_raw_features(x)
        return self.project_raw_features(raw_feats)


class ModalityQualityGating(nn.Module):
    """Learned gating mechanism balancing sensor vs. visual features with explicit missingness support."""

    def __init__(self, sensor_dim: int = 64, visual_dim: int = 64):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(sensor_dim + visual_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        e_sensor: torch.Tensor,
        e_visual: torch.Tensor,
        mask_visual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # mask_visual: [B, 1] with 1.0 = available, 0.0 = missing
        gate_input = torch.cat([e_sensor, e_visual, mask_visual], dim=-1)
        raw_alpha = self.gate_net(gate_input)  # [B, 1] in [0, 1]
        # If visual modality is missing, effective alpha is strictly 0.0
        effective_alpha = raw_alpha * mask_visual
        gated_visual = effective_alpha * e_visual
        fused = torch.cat([e_sensor, gated_visual, mask_visual], dim=-1)  # [B, 64 + 64 + 1 = 129]
        return fused, effective_alpha


class UnifiedHybridModel(nn.Module):
    """Unified Multimodal Model with learned gating and joint PM2.5 + PM10 prediction."""

    def __init__(
        self,
        sensor_input_size: int = 27,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        projection_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.sensor_encoder = LSTMSensorEncoder(
            input_size=sensor_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            projection_dim=projection_dim,
        )
        self.image_encoder = ResNetImageEncoder(
            projection_dim=projection_dim,
            dropout=dropout,
        )
        self.gating = ModalityQualityGating(
            sensor_dim=projection_dim,
            visual_dim=projection_dim,
        )

        fused_dim = projection_dim * 2 + 1  # 64 (sensor) + 64 (gated visual) + 1 (mask) = 129
        self.regression_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # Target 0: PM2.5, Target 1: PM10
            nn.Softplus(),     # Non-negative, smooth physical concentration
        )
        # Initialize output bias to empirical means (~110 ug/m3 for PM2.5, ~250 ug/m3 for PM10)
        self.regression_head[-2].bias.data[0] = 110.0
        self.regression_head[-2].bias.data[1] = 250.0

    def forward(
        self,
        sensor_x: torch.Tensor,
        image_x: Optional[torch.Tensor] = None,
        image_raw_feats: Optional[torch.Tensor] = None,
        mask_visual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = sensor_x.size(0)
        device = sensor_x.device

        e_sensor = self.sensor_encoder(sensor_x)

        if mask_visual is None:
            if image_x is not None or image_raw_feats is not None:
                mask_visual = torch.ones((batch_size, 1), device=device)
            else:
                mask_visual = torch.zeros((batch_size, 1), device=device)

        if image_raw_feats is not None:
            e_visual = self.image_encoder.project_raw_features(image_raw_feats)
        elif image_x is not None:
            e_visual = self.image_encoder(image_x)
        else:
            e_visual = torch.zeros_like(e_sensor)

        # Apply visual mask
        e_visual = e_visual * mask_visual

        fused, alpha = self.gating(e_sensor, e_visual, mask_visual)
        preds = self.regression_head(fused)  # [B, 2] -> [PM2.5, PM10]

        return preds, alpha
