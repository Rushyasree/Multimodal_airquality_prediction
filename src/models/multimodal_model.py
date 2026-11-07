"""Multimodal spatio-temporal air-quality forecasting model."""

import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class MLPBranch(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MultimodalAirQualityModel(nn.Module):
    """Image + AQ + weather + spatial-temporal -> attention -> LSTM -> PM2.5/PM10."""

    def __init__(
        self,
        aq_dim=6,
        weather_dim=5,
        st_dim=5,
        visual_dim=256,
        numerical_dim=128,
        fusion_dim=256,
        lstm_hidden=256,
        lstm_layers=1,
        dropout=0.2,
        freeze_resnet=True,
        prediction_horizon=1,
    ):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Identity()

        if freeze_resnet:
            for p in backbone.parameters():
                p.requires_grad = False

        self.image_backbone = backbone
        self.image_projection = nn.Sequential(
            nn.Linear(2048, visual_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.aq_branch = MLPBranch(aq_dim, numerical_dim, dropout)
        self.weather_branch = MLPBranch(weather_dim, numerical_dim, dropout)
        self.st_branch = MLPBranch(st_dim, numerical_dim, dropout)

        concat_dim = visual_dim + numerical_dim * 3

        self.fusion_projection = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Element-wise attention/gating over the fused representation.
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )

        self.lstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, prediction_horizon * 2),
        )

        self.prediction_horizon = prediction_horizon

    def forward(self, images, air_quality, weather, spatial_temporal):
        # images: [B, T, 3, H, W]
        b, t, c, h, w = images.shape

        x_img = images.reshape(b * t, c, h, w)
        visual = self.image_backbone(x_img)
        visual = self.image_projection(visual).reshape(b, t, -1)

        aq = self.aq_branch(air_quality)
        weather = self.weather_branch(weather)
        st = self.st_branch(spatial_temporal)

        fused = torch.cat([visual, aq, weather, st], dim=-1)
        fused = self.fusion_projection(fused)

        weights = self.attention(fused)
        fused = fused * weights

        sequence_output, _ = self.lstm(fused)
        last_hidden = sequence_output[:, -1, :]

        output = self.head(last_hidden)
        return output.view(b, self.prediction_horizon, 2)
