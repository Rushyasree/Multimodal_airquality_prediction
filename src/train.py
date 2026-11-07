"""Training entry point.

This file intentionally expects a processed aligned dataset. The alignment
pipeline is implemented separately so data assumptions can be inspected and
validated before model training.
"""

from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import MultimodalSequenceDataset
from src.models.multimodal_model import MultimodalAirQualityModel
from src.utils.reproducibility import set_seed


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0

    for batch in loader:
        images = batch["images"].to(device)
        aq = batch["air_quality"].to(device)
        weather = batch["weather"].to(device)
        st = batch["spatial_temporal"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        pred = model(images, aq, weather, st)

        # target [B, 2] -> [B, 1, 2]
        if target.ndim == 2:
            target = target.unsqueeze(1)

        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total += loss.item()

    return total / max(1, len(loader))


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = Path("data/processed/train.pt")
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. First create the real aligned dataset."
        )

    dataset = MultimodalSequenceDataset(data_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = MultimodalAirQualityModel(prediction_horizon=1).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-5,
    )
    criterion = nn.MSELoss()

    for epoch in range(20):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1:02d}/20 | loss={loss:.6f}")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/multimodal_air_quality.pt")
    print("Saved: models/multimodal_air_quality.pt")


if __name__ == "__main__":
    main()
