"""Architecture smoke test using synthetic data only."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from src.models.multimodal_model import MultimodalAirQualityModel

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultimodalAirQualityModel(
        aq_dim=6, weather_dim=5, st_dim=5,
        prediction_horizon=1
    ).to(device)

    B, T = 2, 4
    images = torch.randn(B, T, 3, 224, 224, device=device)
    aq = torch.randn(B, T, 6, device=device)
    weather = torch.randn(B, T, 5, device=device)
    st = torch.randn(B, T, 5, device=device)

    with torch.no_grad():
        y = model(images, aq, weather, st)

    print("Device:", device)
    print("Output shape:", tuple(y.shape))
    print("Expected: (2, 1, 2)")
    assert y.shape == (B, 1, 2)
    print("SMOKE TEST PASSED")

if __name__ == "__main__":
    main()
