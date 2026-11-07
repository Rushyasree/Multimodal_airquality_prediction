
from pathlib import Path
import numpy as np
import pandas as pd
import json

DATA_FILE = Path("data/processed/air_quality_weather_merged.csv")
SEQUENCE_FILE = Path(
    "data/processed/sequences/air_quality_24h_sequence_index.csv"
)
MODEL_DIR = Path("models/lstm_baseline")

def metrics(y_true, y_pred):
    results = {}
    for j, name in enumerate(["PM2.5", "PM10"]):
        error = y_pred[:, j] - y_true[:, j]
        ss_res = np.sum(error ** 2)
        ss_tot = np.sum((y_true[:, j] - y_true[:, j].mean()) ** 2)

        results[name] = {
            "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
        }
    return results

def main():
    print("=" * 68)
    print("PERSISTENCE BASELINE VS LSTM")
    print("=" * 68)

    merged = pd.read_csv(DATA_FILE, low_memory=False)
    seq = pd.read_csv(SEQUENCE_FILE)

    merged["StationId"] = (
        merged["StationId"].astype(str).str.strip().str.upper()
    )
    seq["StationId"] = (
        seq["StationId"].astype(str).str.strip().str.upper()
    )

    merged["Datetime"] = pd.to_datetime(merged["Datetime"])
    for col in ["history_end", "target_time"]:
        seq[col] = pd.to_datetime(seq[col])

    test_seq = (
        seq.loc[seq["split"].str.lower() == "test"]
        .sort_values(["StationId", "target_time"])
        .reset_index(drop=True)
    )

    # The sequence's final observed hour supplies persistence forecasts.
    last_hour = merged[
        ["StationId", "Datetime", "PM2.5", "PM10"]
    ].rename(
        columns={
            "Datetime": "history_end",
            "PM2.5": "persistence_PM2.5",
            "PM10": "persistence_PM10",
        }
    )

    test = test_seq.merge(
        last_hour,
        on=["StationId", "history_end"],
        how="left",
        validate="one_to_one",
    )

    if test[["persistence_PM2.5", "persistence_PM10"]].isna().any().any():
        raise ValueError("Missing observations for persistence baseline.")

    y_true = test[["target_PM2.5", "target_PM10"]].to_numpy(
        dtype=np.float64
    )
    y_persistence = test[
        ["persistence_PM2.5", "persistence_PM10"]
    ].to_numpy(dtype=np.float64)

    prediction_path = MODEL_DIR / "test_predictions.npy"
    y_test_path = Path("data/processed/model_sequences/y_test.npy")

    y_lstm = np.load(prediction_path).astype(np.float64)
    y_saved = np.load(y_test_path).astype(np.float64)

    if len(y_lstm) != len(test) or len(y_saved) != len(test):
        raise ValueError("Test predictions and test sequence counts differ.")

    # Verify the order and targets match the generated test arrays.
    if not np.allclose(y_saved, y_true, rtol=1e-5, atol=1e-4):
        raise ValueError(
            "Test sequence labels do not match y_test.npy order. "
            "Do not compare models until alignment is corrected."
        )

    persistence_metrics = metrics(y_true, y_persistence)
    lstm_metrics = metrics(y_true, y_lstm)

    print(f"\nTest samples: {len(test):,}")

    for target in ["PM2.5", "PM10"]:
        print(f"\n{target}")
        for model_name, result in [
            ("Persistence", persistence_metrics[target]),
            ("LSTM", lstm_metrics[target]),
        ]:
            print(
                f"{model_name:12s} | "
                f"MAE={result['MAE']:.4f} | "
                f"RMSE={result['RMSE']:.4f} | "
                f"R2={result['R2']:.4f}"
            )

        improvement = (
            persistence_metrics[target]["MAE"]
            - lstm_metrics[target]["MAE"]
        )
        print(f"LSTM MAE improvement: {improvement:.4f}")

    report = {
        "test_samples": len(test),
        "persistence": persistence_metrics,
        "lstm": lstm_metrics,
        "note": "One-hour persistence baseline; same held-out test targets.",
    }

    output = MODEL_DIR / "persistence_comparison.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved comparison: {output}")

if __name__ == "__main__":
    main()
