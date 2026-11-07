"""
experiments/unified_hybrid/accuracy_optimization/hybrid_model_optimized.py

Optimized Residual Quality-Aware Multimodal Fusion Architecture:
1. Base Sensor Forecaster: Pretrained 2-layer LSTM + regressor with full weights intact.
2. Visual Feature Projector: Projects 2048-dim ResNet-50 features into a low-rank representation.
3. Quality-Aware Adaptive Gating: Gate alpha in [0, 1] conditioned on sensor, vision, distance, and mask.
4. Bounded Residual Head: Predicts bounded physical adjustments (delta_pm25, delta_pm10).
5. Exact Fallback Guarantee: When visual mask = 0, output defaults identically to the pretrained sensor forecast.
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import numpy as np


class PretrainedLSTMWrapper(nn.Module):
    """Wraps the trained 2-layer LSTM forecaster to generate baseline predictions

    and intermediate hidden representations.
    """

    def __init__(
        self,
        checkpoint_path: str,
        freeze: bool = True,
    ):
        super().__init__()
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        input_size = ckpt.get("input_size", 27)
        hidden_size = ckpt.get("hidden_size", 64)
        num_layers = ckpt.get("num_layers", 2)
        dropout = ckpt.get("dropout", 0.2)
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
        
        # Load state dict
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.load_state_dict(sd)
        
        # Target scaling parameters
        t_mean = ckpt.get("target_mean", [79.5145, 172.1068])
        t_std = ckpt.get("target_std", [77.4174, 137.4705])
        self.register_buffer("target_mean", torch.tensor(t_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(t_std, dtype=torch.float32))
        
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns:

        - y_sensor_physical: [B, 2] physical predictions in ug/m3
        - y_sensor_norm: [B, 2] normalized predictions
        - last_hidden: [B, 64] final LSTM hidden state
        """
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        y_sensor_norm = self.regressor(last_hidden)
        y_sensor_physical = y_sensor_norm * self.target_std + self.target_mean
        return y_sensor_physical, y_sensor_norm, last_hidden


class VisualFeatureProjector(nn.Module):
    """Projects 2048-dim ResNet-50 visual features into a regularized representation."""

    def __init__(self, in_features: int = 2048, out_features: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QualityAwareResidualHead(nn.Module):
    """Computes quality-conditioned adaptive gate alpha and bounded residual delta."""

    def __init__(
        self,
        sensor_dim: int = 64,
        visual_dim: int = 32,
        dist_dim: int = 8,
        hidden_dim: int = 32,
        max_delta_pm25: float = 30.0,
        max_delta_pm10: float = 50.0,
    ):
        super().__init__()
        self.max_delta_pm25 = max_delta_pm25
        self.max_delta_pm10 = max_delta_pm10
        
        # Spatial distance embedder
        self.dist_embedder = nn.Sequential(
            nn.Linear(1, dist_dim),
            nn.ReLU(),
        )
        
        fused_in = sensor_dim + visual_dim + dist_dim
        
        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(fused_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # Residual network predicting delta
        self.residual_net = nn.Sequential(
            nn.Linear(fused_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),  # Bounded in [-1, 1]
        )
        
        # Initialize residual output layer near zero so initial delta is zero
        nn.init.zeros_(self.residual_net[-2].weight)
        nn.init.zeros_(self.residual_net[-2].bias)

    def forward(
        self,
        h_sensor: torch.Tensor,
        e_visual: torch.Tensor,
        distance_km: torch.Tensor,
        mask_visual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns:

        - delta: [B, 2] bounded residual correction in ug/m3
        - alpha: [B, 1] effective gating factor in [0, 1]
        """
        e_dist = self.dist_embedder(distance_km)
        
        # When visual mask is 0, zero out visual features
        e_visual_effective = e_visual * mask_visual
        
        fused = torch.cat([h_sensor, e_visual_effective, e_dist], dim=-1)
        
        raw_alpha = self.gate_net(fused)
        effective_alpha = raw_alpha * mask_visual  # Strictly 0 if visual modality is absent
        
        raw_delta = self.residual_net(fused)  # [-1, 1]
        
        # Scale to max allowable residual bounds
        delta_pm25 = raw_delta[:, 0:1] * self.max_delta_pm25
        delta_pm10 = raw_delta[:, 1:2] * self.max_delta_pm10
        delta = torch.cat([delta_pm25, delta_pm10], dim=-1)
        
        return delta, effective_alpha


class OptimizedResidualHybridModel(nn.Module):
    """Optimized Residual Quality-Aware Multimodal Fusion Model.

    Combines pretrained sensor forecaster with bounded visual residual correction.
    """

    def __init__(
        self,
        lstm_checkpoint_path: str,
        freeze_sensor: bool = True,
        visual_dim: int = 32,
        dropout: float = 0.2,
        max_delta_pm25: float = 25.0,
        max_delta_pm10: float = 40.0,
    ):
        super().__init__()
        self.sensor_forecaster = PretrainedLSTMWrapper(lstm_checkpoint_path, freeze=freeze_sensor)
        self.visual_projector = VisualFeatureProjector(in_features=2048, out_features=visual_dim, dropout=dropout)
        self.quality_residual_head = QualityAwareResidualHead(
            sensor_dim=64,
            visual_dim=visual_dim,
            dist_dim=8,
            hidden_dim=32,
            max_delta_pm25=max_delta_pm25,
            max_delta_pm10=max_delta_pm10,
        )

    def forward(
        self,
        sensor_x: torch.Tensor,
        image_feats: Optional[torch.Tensor] = None,
        distance_km: Optional[torch.Tensor] = None,
        mask_visual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = sensor_x.size(0)
        device = sensor_x.device
        
        # 1. Base Sensor Prediction
        y_sensor, y_sensor_norm, h_sensor = self.sensor_forecaster(sensor_x)
        
        # 2. Defaults for optional arguments
        if mask_visual is None:
            mask_visual = torch.ones((batch_size, 1), device=device) if image_feats is not None else torch.zeros((batch_size, 1), device=device)
        
        if distance_km is None:
            distance_km = torch.full((batch_size, 1), 0.75, device=device)
            
        if image_feats is None:
            image_feats = torch.zeros((batch_size, 2048), device=device)
            mask_visual = torch.zeros((batch_size, 1), device=device)
            
        # 3. Visual feature projection
        e_visual = self.visual_projector(image_feats)
        
        # 4. Residual computation and quality gating
        delta, alpha = self.quality_residual_head(h_sensor, e_visual, distance_km, mask_visual)
        
        # 5. Final residual hybrid prediction with physical non-negativity bound
        y_hybrid = torch.clamp(y_sensor + alpha * delta, min=0.0)
        
        details = {
            "y_sensor": y_sensor,
            "delta": delta,
            "alpha": alpha,
            "mask_visual": mask_visual,
        }
        
        return y_hybrid, alpha, details
