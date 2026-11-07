"""
experiments/multimodal_fusion/fusion_model.py

PyTorch implementations of:
1. LSTMSensorEncoder (24-hour sequence, 27 features)
2. ResNetImageEncoder (224x224 RGB image, ResNet-50 backbone)
3. MultimodalFusionModel (Feature-level concatenation + MLP regression head)
4. SensorOnlyModel (Sensor baseline)
5. ImageOnlyModel (Image baseline)

Predicts continuous, non-negative PM2.5 in physical units (ug/m3) via smooth Softplus activation.
"""

import torch
import torch.nn as nn
from torchvision import models


class LSTMSensorEncoder(nn.Module):
    """Encodes 24-hour sensor sequence (24x27) into a dense embedding."""

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
        """Loads compatible LSTM layer weights from best_lstm_model.pt."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt["model_state_dict"]
        lstm_sd = {k: v for k, v in sd.items() if k.startswith("lstm.")}
        self.load_state_dict(lstm_sd, strict=False)
        print(f"[SensorEncoder] Successfully loaded {len(lstm_sd)} LSTM tensors from {checkpoint_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 24, 27]
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]  # [B, 64]
        return self.projection(last_hidden)  # [B, projection_dim]


class ResNetImageEncoder(nn.Module):
    """Encodes 224x224 RGB camera image into a dense embedding using ResNet-50."""

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
        """Loads compatible ResNet-50 backbone weights from best_model.pt."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt["model_state_dict"]
        backbone_sd = {k: v for k, v in sd.items() if k.startswith("backbone.")}
        self.load_state_dict(backbone_sd, strict=False)
        print(f"[ImageEncoder] Successfully loaded {len(backbone_sd)} backbone tensors from {checkpoint_path}")

    def extract_raw_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 2048-dim global average pooled features."""
        feats = self.backbone(x)
        return torch.flatten(feats, 1)  # [B, 2048]

    def project_raw_features(self, raw_feats: torch.Tensor) -> torch.Tensor:
        """Projects 2048-dim features to projection_dim."""
        return self.projection(raw_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_feats = self.extract_raw_features(x)
        return self.project_raw_features(raw_feats)


class MultimodalFusionModel(nn.Module):
    """Fused architecture: Concatenates sensor and image embeddings + MLP head."""

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
        
        fused_dim = projection_dim * 2  # Sensor (64) + Image (64) = 128
        self.regression_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # Smooth, non-negative, non-saturating
        )
        # Initialize output bias to empirical mean ~100 ug/m3
        self.regression_head[-2].bias.data.fill_(100.0)

    def forward(
        self, 
        sensor_x: torch.Tensor, 
        image_x: torch.Tensor = None, 
        image_raw_feats: torch.Tensor = None
    ) -> torch.Tensor:
        e_sensor = self.sensor_encoder(sensor_x)
        if image_raw_feats is not None:
            e_image = self.image_encoder.project_raw_features(image_raw_feats)
        else:
            e_image = self.image_encoder(image_x)
            
        fused = torch.cat([e_sensor, e_image], dim=-1)
        return self.regression_head(fused)


class SensorOnlyModel(nn.Module):
    """Sensor-only baseline model for controlled ablation."""

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
        self.regression_head = nn.Sequential(
            nn.Linear(projection_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        self.regression_head[-2].bias.data.fill_(100.0)

    def forward(self, sensor_x: torch.Tensor) -> torch.Tensor:
        e_sensor = self.sensor_encoder(sensor_x)
        return self.regression_head(e_sensor)


class ImageOnlyModel(nn.Module):
    """Image-only baseline model for controlled ablation."""

    def __init__(
        self,
        projection_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.image_encoder = ResNetImageEncoder(
            projection_dim=projection_dim,
            dropout=dropout,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(projection_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        self.regression_head[-2].bias.data.fill_(100.0)

    def forward(
        self, 
        image_x: torch.Tensor = None, 
        image_raw_feats: torch.Tensor = None
    ) -> torch.Tensor:
        if image_raw_feats is not None:
            e_image = self.image_encoder.project_raw_features(image_raw_feats)
        else:
            e_image = self.image_encoder(image_x)
        return self.regression_head(e_image)
