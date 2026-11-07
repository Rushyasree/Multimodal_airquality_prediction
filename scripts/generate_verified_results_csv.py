import pandas as pd
from pathlib import Path

ROOT = Path("E:/Documents/multimodal_air_quality_vscode_starter")
OUT_PATH = ROOT / "results" / "multimodal" / "verified_results.csv"

rows = [
    # Category 1: Numerical Next-Hour Forecasting
    {
        "Experiment_Category": "Numerical Forecasting (Operational Scale)",
        "Model": "2-Layer LSTM Forecaster",
        "Modality": "Numerical (Pollution + ERA5 Weather + Time)",
        "Target": "PM2.5",
        "Forecast_Horizon": "Next-hour (t+1)",
        "Test_Sample_Count": 404431,
        "Independent_Test_Groups": "Held-out chronological test period across 90 stations",
        "MAE": 10.5719,
        "RMSE": 18.5789,
        "R2": 0.9492,
        "Split_Strategy": "Chronological train/val/test split (2015-2018 train, 2018-2019 val, 2019-2020 test)",
        "Limitations": "Requires continuous hourly sensor telemetry; degrades under prolonged missing-data periods."
    },
    {
        "Experiment_Category": "Numerical Forecasting (Operational Scale)",
        "Model": "2-Layer LSTM Forecaster",
        "Modality": "Numerical (Pollution + ERA5 Weather + Time)",
        "Target": "PM10",
        "Forecast_Horizon": "Next-hour (t+1)",
        "Test_Sample_Count": 404431,
        "Independent_Test_Groups": "Held-out chronological test period across 90 stations",
        "MAE": 18.8643,
        "RMSE": 32.2027,
        "R2": 0.9301,
        "Split_Strategy": "Chronological train/val/test split (2015-2018 train, 2018-2019 val, 2019-2020 test)",
        "Limitations": "Requires continuous hourly sensor telemetry; degrades under prolonged missing-data periods."
    },
    {
        "Experiment_Category": "Numerical Forecasting (Operational Scale)",
        "Model": "Persistence Baseline",
        "Modality": "Numerical (Lagged PM t)",
        "Target": "PM2.5",
        "Forecast_Horizon": "Next-hour (t+1)",
        "Test_Sample_Count": 404431,
        "Independent_Test_Groups": "Held-out chronological test period across 90 stations",
        "MAE": 10.6783,
        "RMSE": 19.8072,
        "R2": 0.9423,
        "Split_Strategy": "Identical chronological test partition as LSTM",
        "Limitations": "Naive benchmark assuming no change (y_hat = y_t); cannot predict sudden pollution spikes or diurnal transitions."
    },
    {
        "Experiment_Category": "Numerical Forecasting (Operational Scale)",
        "Model": "Persistence Baseline",
        "Modality": "Numerical (Lagged PM t)",
        "Target": "PM10",
        "Forecast_Horizon": "Next-hour (t+1)",
        "Test_Sample_Count": 404431,
        "Independent_Test_Groups": "Held-out chronological test period across 90 stations",
        "MAE": 19.7384,
        "RMSE": 36.2374,
        "R2": 0.9115,
        "Split_Strategy": "Identical chronological test partition as LSTM",
        "Limitations": "Naive benchmark assuming no change (y_hat = y_t); cannot predict sudden pollution spikes or diurnal transitions."
    },

    # Category 2: Image-Only Environmental Vision
    {
        "Experiment_Category": "Image-Only Environmental Vision",
        "Model": "ResNet-50 Transfer Learning",
        "Modality": "Image-only (PM25Vision)",
        "Target": "Image-associated PM2.5",
        "Forecast_Horizon": "Snapshot estimation (Nowcasting)",
        "Test_Sample_Count": 79,
        "Independent_Test_Groups": "18 station-date groups across 13 stations",
        "MAE": 66.2148,
        "RMSE": 80.6779,
        "R2": 0.1524,
        "Split_Strategy": "GroupShuffleSplit by station-date (439 train / 122 val / 79 test; 640 unique images total)",
        "Limitations": "No sub-daily timestamp; sensitive to illumination, camera optics, and background scene variation."
    },

    # Category 3: Exploratory Daily Matched Cohort Analysis
    {
        "Experiment_Category": "Exploratory Daily Matched Cohort Analysis",
        "Model": "Numerical Ridge Regressor",
        "Modality": "Numerical (Daily ERA5 Weather)",
        "Target": "Station Daily Mean PM2.5",
        "Forecast_Horizon": "Daily mean estimation",
        "Test_Sample_Count": 60,
        "Independent_Test_Groups": "9 station-date groups across held-out stations",
        "MAE": 193.1787,
        "RMSE": 219.8654,
        "R2": -5.4129,
        "Split_Strategy": "GroupShuffleSplit on 36 matched station-date groups (205 train / 60 test; 265 samples total)",
        "Limitations": "Severely restricted sample (27 train groups); daily meteorology alone cannot infer spatial baseline pollution."
    },
    {
        "Experiment_Category": "Exploratory Daily Matched Cohort Analysis",
        "Model": "Image Ridge Regressor",
        "Modality": "Image-only (ResNet-50 Features)",
        "Target": "Station Daily Mean PM2.5",
        "Forecast_Horizon": "Daily mean estimation",
        "Test_Sample_Count": 60,
        "Independent_Test_Groups": "9 station-date groups across held-out stations",
        "MAE": 73.6699,
        "RMSE": 95.3836,
        "R2": -0.2070,
        "Split_Strategy": "GroupShuffleSplit on 36 matched station-date groups (205 train / 60 test; 265 samples total)",
        "Limitations": "Negative R2 demonstrates out-of-distribution domain shift across unseen urban stations."
    },
    {
        "Experiment_Category": "Exploratory Daily Matched Cohort Analysis",
        "Model": "Multimodal Early Fusion (Ridge)",
        "Modality": "Multimodal (ResNet-50 + Weather)",
        "Target": "Station Daily Mean PM2.5",
        "Forecast_Horizon": "Daily mean estimation",
        "Test_Sample_Count": 60,
        "Independent_Test_Groups": "9 station-date groups across held-out stations",
        "MAE": 85.8619,
        "RMSE": 106.5622,
        "R2": -0.5064,
        "Split_Strategy": "GroupShuffleSplit on 36 matched station-date groups (205 train / 60 test; 265 samples total)",
        "Limitations": "Performs worse than image-only (MAE 85.86 vs 73.67); multimodal features overfit the small 27-group training set."
    },
    {
        "Experiment_Category": "Exploratory Daily Matched Cohort Analysis",
        "Model": "Numerical Ridge Regressor",
        "Modality": "Numerical (Daily ERA5 Weather)",
        "Target": "Station Daily Mean PM10",
        "Forecast_Horizon": "Daily mean estimation",
        "Test_Sample_Count": 60,
        "Independent_Test_Groups": "9 station-date groups across held-out stations",
        "MAE": 297.1218,
        "RMSE": 349.2314,
        "R2": -3.2755,
        "Split_Strategy": "GroupShuffleSplit on 36 matched station-date groups (205 train / 60 test; 265 samples total)",
        "Limitations": "Severely restricted sample (27 train groups); daily meteorology alone cannot infer spatial baseline pollution."
    },
    {
        "Experiment_Category": "Exploratory Daily Matched Cohort Analysis",
        "Model": "Image Ridge Regressor",
        "Modality": "Image-only (ResNet-50 Features)",
        "Target": "Station Daily Mean PM10",
        "Forecast_Horizon": "Daily mean estimation",
        "Test_Sample_Count": 60,
        "Independent_Test_Groups": "9 station-date groups across held-out stations",
        "MAE": 137.6515,
        "RMSE": 172.1311,
        "R2": -0.0387,
        "Split_Strategy": "GroupShuffleSplit on 36 matched station-date groups (205 train / 60 test; 265 samples total)",
        "Limitations": "Negative R2 demonstrates out-of-distribution domain shift across unseen urban stations."
    },
    {
        "Experiment_Category": "Exploratory Daily Matched Cohort Analysis",
        "Model": "Multimodal Early Fusion (Ridge)",
        "Modality": "Multimodal (ResNet-50 + Weather)",
        "Target": "Station Daily Mean PM10",
        "Forecast_Horizon": "Daily mean estimation",
        "Test_Sample_Count": 60,
        "Independent_Test_Groups": "9 station-date groups across held-out stations",
        "MAE": 150.4686,
        "RMSE": 186.1649,
        "R2": -0.2150,
        "Split_Strategy": "GroupShuffleSplit on 36 matched station-date groups (205 train / 60 test; 265 samples total)",
        "Limitations": "Performs worse than image-only (MAE 150.47 vs 137.65); multimodal features overfit the small 27-group training set."
    }
]

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)
print(f"Saved verified results table to: {OUT_PATH}")
