"""Evaluation entry point for the trained model."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MultimodalSequenceDataset
from src.models.multimodal_model import MultimodalAirQualityModel
from src.utils.metrics import regression_metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path("data/processed/test.pt")
    model_path = Path("models/multimodal_air_quality.pt")

    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found.")
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found.")

    dataset = MultimodalSequenceDataset(data_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = MultimodalAirQualityModel(prediction_horizon=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["images"].to(device),
                batch["air_quality"].to(device),
                batch["weather"].to(device),
                batch["spatial_temporal"].to(device),
            )
            ys.append(batch["target"].numpy())
            ps.append(pred[:, 0, :].cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)

    for i, name in enumerate(["PM2.5", "PM10"]):
        print(name, regression_metrics(y_true[:, i], y_pred[:, i]))


if __name__ == "__main__":
    main()
