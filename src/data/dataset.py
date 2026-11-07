"""Dataset classes for aligned multimodal sequences.

Expected processed sample format:
images:      [T, 3, 224, 224]
air_quality: [T, aq_features]
weather:     [T, weather_features]
spatial_temporal: [T, st_features]
target:      [2] for next-step PM2.5 and PM10
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset


class MultimodalSequenceDataset(Dataset):
    """Load a serialized list of aligned samples."""

    def __init__(self, samples_path):
        self.samples_path = Path(samples_path)
        self.samples = torch.load(self.samples_path, weights_only=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "images": sample["images"].float(),
            "air_quality": sample["air_quality"].float(),
            "weather": sample["weather"].float(),
            "spatial_temporal": sample["spatial_temporal"].float(),
            "target": sample["target"].float(),
        }
