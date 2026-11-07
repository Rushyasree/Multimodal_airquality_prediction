import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import torch
from torchvision import models, transforms
from PIL import Image

# ---------------------------------------------------------
# SETUP & PATHS
# ---------------------------------------------------------
# Ensure ROOT points to project base directory
current_file = Path(__file__).resolve()
ROOT = current_file.parent.parent if current_file.parent.name in ["scripts", "scratch"] else current_file.parent

OUT_DIR = ROOT / "results" / "multimodal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70)
print(f"GENERATING MULTIMODAL FEASIBILITY AUDIT AND EXPERIMENT ARTIFACTS")
print(f"Project ROOT: {ROOT}")
print(f"Output DIR:   {OUT_DIR}")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD DATA & AUDIT
# ---------------------------------------------------------
manifest = pd.read_csv(ROOT / "data" / "processed" / "pm25vision_cnn" / "pm25vision_clean_manifest.csv")
aq = pd.read_csv(ROOT / "data" / "processed" / "air_quality_weather_merged.csv", low_memory=False)
aq["Datetime"] = pd.to_datetime(aq["Datetime"], errors="coerce")
aq["date"] = aq["Datetime"].dt.strftime("%Y-%m-%d")

# Spatial distances
distances = manifest["distance_to_station_km"]

# Station daily ground truth
daily = aq.groupby(["StationId", "date"]).agg({
    "PM2.5": ["mean", "std", "min", "max", "count"],
    "PM10": ["mean", "std", "min", "max", "count"],
    "temperature_C": "mean",
    "dewpoint_C": "mean",
    "relative_humidity": "mean",
    "wind_speed": "mean",
    "precipitation_mm": "mean",
})
daily.columns = ["_".join(c).strip("_") for c in daily.columns]
daily = daily.reset_index()

within_5km = manifest[manifest["within_threshold"]].copy()
matched = pd.merge(
    within_5km,
    daily,
    left_on=["nearest_station_id", "capture_date_clean"],
    right_on=["StationId", "date"],
    how="inner"
).reset_index(drop=True)

diff_pm25 = matched["pm25_label"] - matched["PM2.5_mean"]
group_counts = matched["station_date_group"].value_counts()

# ---------------------------------------------------------
# 2. RUN CONTROLLED MATCHED EXPERIMENT
# ---------------------------------------------------------
print("Extracting ResNet-50 features for matched images...")
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

weights = models.ResNet50_Weights.DEFAULT
backbone = models.resnet50(weights=weights)
feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-1])
feature_extractor.eval()

img_feats = []
with torch.no_grad():
    for p in matched["image_path"]:
        img_path = ROOT / p if not Path(p).is_absolute() else Path(p)
        with Image.open(img_path) as img:
            t = val_transform(img.convert("RGB")).unsqueeze(0)
            f = torch.flatten(feature_extractor(t), 1).squeeze(0).numpy()
            img_feats.append(f)

X_img = np.array(img_feats)
num_cols = ["temperature_C_mean", "dewpoint_C_mean", "relative_humidity_mean", "wind_speed_mean", "precipitation_mm_mean"]
X_num = matched[num_cols].fillna(matched[num_cols].median()).values

y_pm25 = matched["PM2.5_mean"].values
y_pm10 = matched["PM10_mean"].values

# Group-safe splitting
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
train_idx, test_idx = next(gss.split(matched, groups=matched["station_date_group"]))

train_groups = set(matched.loc[train_idx, "station_date_group"])
test_groups = set(matched.loc[test_idx, "station_date_group"])
assert train_groups.isdisjoint(test_groups)

scaler_num = StandardScaler()
X_num_tr = scaler_num.fit_transform(X_num[train_idx])
X_num_te = scaler_num.transform(X_num[test_idx])

scaler_img = StandardScaler()
X_img_tr = scaler_img.fit_transform(X_img[train_idx])
X_img_te = scaler_img.transform(X_img[test_idx])

X_fus_tr = np.hstack([X_img_tr, X_num_tr])
X_fus_te = np.hstack([X_img_te, X_num_te])

y_tr_pm25 = y_pm25[train_idx]
y_te_pm25 = y_pm25[test_idx]
y_tr_pm10 = y_pm10[train_idx]
y_te_pm10 = y_pm10[test_idx]

# Train Ridge regressors
# PM2.5
reg_num_pm25 = Ridge(alpha=10.0).fit(X_num_tr, y_tr_pm25)
pred_num_pm25 = reg_num_pm25.predict(X_num_te)

reg_img_pm25 = Ridge(alpha=100.0).fit(X_img_tr, y_tr_pm25)
pred_img_pm25 = reg_img_pm25.predict(X_img_te)

reg_fus_pm25 = Ridge(alpha=100.0).fit(X_fus_tr, y_tr_pm25)
pred_fus_pm25 = reg_fus_pm25.predict(X_fus_te)

# PM10
v_tr10 = ~np.isnan(y_tr_pm10)
v_te10 = ~np.isnan(y_te_pm10)

reg_num_pm10 = Ridge(alpha=10.0).fit(X_num_tr[v_tr10], y_tr_pm10[v_tr10])
pred_num_pm10 = reg_num_pm10.predict(X_num_te[v_te10])

reg_img_pm10 = Ridge(alpha=100.0).fit(X_img_tr[v_tr10], y_tr_pm10[v_tr10])
pred_img_pm10 = reg_img_pm10.predict(X_img_te[v_te10])

reg_fus_pm10 = Ridge(alpha=100.0).fit(X_fus_tr[v_tr10], y_tr_pm10[v_tr10])
pred_fus_pm10 = reg_fus_pm10.predict(X_fus_te[v_te10])

def eval_m(y_true, y_p):
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_p)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_p))), 4),
        "R2": round(float(r2_score(y_true, y_p)), 4),
    }

m_num_25 = eval_m(y_te_pm25, pred_num_pm25)
m_img_25 = eval_m(y_te_pm25, pred_img_pm25)
m_fus_25 = eval_m(y_te_pm25, pred_fus_pm25)

m_num_10 = eval_m(y_te_pm10[v_te10], pred_num_pm10)
m_img_10 = eval_m(y_te_pm10[v_te10], pred_img_pm10)
m_fus_10 = eval_m(y_te_pm10[v_te10], pred_fus_pm10)

# ---------------------------------------------------------
# 3. WRITE FEASIBILITY REPORT JSON
# ---------------------------------------------------------
feasibility_report = {
    "project": "Multimodal Spatio-Temporal Air Quality Prediction for Indian Urban Environments",
    "feasibility_assessment": {
        "is_hourly_multimodal_fusion_defensible": False,
        "is_exploratory_matched_analysis_defensible": True,
        "primary_reasons": [
            "Lack of sub-daily timestamps: PM25Vision records dates only (YYYY-MM-DD), with no hour/minute data, making alignment with 24-hour sequence forecasting impossible without arbitrary hour assignment.",
            "Extreme intraday variance: Station PM2.5 fluctuates by an average of 166.72 ug/m3 (min 85.21 to max 251.93 ug/m3) within a single day at matched sites.",
            "Severe effective sample size bottleneck: Only 265 image-station pairs share a date within 5 km, representing only 36 independent station-date groups across 19 stations over 5.5 years.",
            "Heavy spatial and temporal clustering: 40% of all matched images originate from just two bursts (63 images at DL002 on 2016-01-30 and 42 images at DL026 on 2018-10-24).",
            "Target discrepancy and bias: PM25Vision image labels have an average positive bias of +70.74 ug/m3 relative to CPCB ground monitors (236.83 vs 166.09 ug/m3), reflecting distinct underlying data generation processes."
        ]
    },
    "phase_2_audit_answers": {
        "q1_sample_sizes": {
            "clean_manifest_rows": len(manifest),
            "unique_image_ids": int(manifest["image_id_clean"].nunique()),
            "unique_sha256_contents": int(manifest["sha256"].nunique()),
            "unique_nearest_stations_all": int(manifest["nearest_station_id"].nunique()),
            "unique_station_date_groups_all": int(manifest["station_date_group"].nunique()),
            "temporal_coverage": f"{manifest['capture_date_clean'].min()} to {manifest['capture_date_clean'].max()}"
        },
        "q2_spatial_and_temporal_matching": {
            "cpcb_stations_evaluated": 90,
            "threshold_km": 5.0,
            "images_within_5km": int((distances <= 5.0).sum()),
            "images_exceeding_5km": int((distances > 5.0).sum()),
            "distance_min_km": round(float(distances.min()), 4),
            "distance_median_km": round(float(distances.median()), 4),
            "distance_mean_km": round(float(distances.mean()), 4),
            "distance_max_km": round(float(distances.max()), 4),
            "images_within_1km": int((distances <= 1.0).sum()),
            "images_within_2km": int((distances <= 2.0).sum())
        },
        "q3_independent_matched_groups": {
            "matched_images_count": len(matched),
            "unique_station_date_groups": int(matched["station_date_group"].nunique()),
            "unique_stations_matched": int(matched["nearest_station_id"].nunique()),
            "group_size_mean": round(float(group_counts.mean()), 2),
            "group_size_median": float(group_counts.median()),
            "group_size_max": int(group_counts.max()),
            "top_group": f"{group_counts.index[0]} ({group_counts.iloc[0]} images)",
            "second_group": f"{group_counts.index[1]} ({group_counts.iloc[1]} images)"
        },
        "q4_repeated_images_analysis": {
            "finding": "Repeated images in a station-date group represent distinct camera captures (distinct SHA-256 hashes) taken in rapid succession at nearly identical GPS coordinates, all sharing the same static PM2.5 label.",
            "example_dl002": {
                "group": "DL002__2016-01-30",
                "images_count": 63,
                "unique_pm25_labels": [394.0],
                "lat_range": [28.6411, 28.6514],
                "lon_range": [77.3159, 77.3203],
                "distance_range_km": [0.309, 0.787],
                "unique_sha256": 63
            }
        },
        "q5_task_support_assessment": {
            "next_hour_forecasting_supported": False,
            "same_day_estimation_supported": "Exploratory only (small N)",
            "exploratory_image_only_supported": True,
            "intraday_station_pm25_min_mean": round(float(matched["PM2.5_min"].mean()), 2),
            "intraday_station_pm25_max_mean": round(float(matched["PM2.5_max"].mean()), 2),
            "intraday_pm25_range_mean": round(float((matched["PM2.5_max"] - matched["PM2.5_min"]).mean()), 2)
        },
        "q6_statistical_power_and_splits": {
            "total_independent_groups": int(matched["station_date_group"].nunique()),
            "train_groups_count": len(train_groups),
            "test_groups_count": len(test_groups),
            "train_samples_count": len(train_idx),
            "test_samples_count": len(test_idx),
            "test_pm25_mean": round(float(np.mean(y_te_pm25)), 2),
            "test_pm25_std": round(float(np.std(y_te_pm25)), 2)
        }
    },
    "controlled_matched_experiment_metrics": {
        "PM2.5": {
            "Numerical_Only": m_num_25,
            "Image_Only": m_img_25,
            "Multimodal_Fusion": m_fus_25
        },
        "PM10": {
            "Numerical_Only": m_num_10,
            "Image_Only": m_img_10,
            "Multimodal_Fusion": m_fus_10
        }
    },
    "recommended_paper_strategy": {
        "main_forecasting_model": "Numerical LSTM (404,431 test sequences, PM2.5 MAE=10.57, R2=0.949)",
        "vision_model_role": "Image-only exploratory baseline (ResNet-50 transfer learning, 640 images, Test MAE=66.21, R2=0.152)",
        "fusion_conclusion": "Rigorous negative result / boundary condition: Opportunistic ground imagery lacking precise sub-daily timestamps cannot defensibly enhance continuous hourly forecasting systems. Future multimodal architectures require synchronized sensor-camera capture networks."
    }
}

with open(OUT_DIR / "feasibility_report.json", "w", encoding="utf-8") as f:
    json.dump(feasibility_report, f, indent=2)
print("Saved feasibility_report.json")

# ---------------------------------------------------------
# 4. WRITE EXPERIMENT SUMMARY CSV
# ---------------------------------------------------------
summary_rows = [
    {
        "Experiment_Type": "Primary Forecasting Baseline",
        "Model": "LSTM Forecaster (2-layer)",
        "Modality": "Numerical (Pollution + Weather + Time)",
        "Target_Task": "Next-hour (t+1) Forecasting",
        "Target_Pollutant": "PM2.5",
        "Test_Samples": 404431,
        "Independent_Test_Groups": "Held-out temporal split",
        "MAE": 10.5719,
        "RMSE": 18.5789,
        "R2": 0.9492,
        "Evaluation_Notes": "Primary research baseline on 1.37M hourly CPCB/ERA5 records."
    },
    {
        "Experiment_Type": "Primary Forecasting Baseline",
        "Model": "LSTM Forecaster (2-layer)",
        "Modality": "Numerical (Pollution + Weather + Time)",
        "Target_Task": "Next-hour (t+1) Forecasting",
        "Target_Pollutant": "PM10",
        "Test_Samples": 404431,
        "Independent_Test_Groups": "Held-out temporal split",
        "MAE": 18.8643,
        "RMSE": 32.2027,
        "R2": 0.9301,
        "Evaluation_Notes": "Primary research baseline on 1.37M hourly CPCB/ERA5 records."
    },
    {
        "Experiment_Type": "Reference Baseline",
        "Model": "Persistence Baseline",
        "Modality": "Numerical (Lagged PM t)",
        "Target_Task": "Next-hour (t+1) Forecasting",
        "Target_Pollutant": "PM2.5",
        "Test_Samples": 404431,
        "Independent_Test_Groups": "Held-out temporal split",
        "MAE": 10.6783,
        "RMSE": 19.8072,
        "R2": 0.9423,
        "Evaluation_Notes": "Standard persistence benchmark."
    },
    {
        "Experiment_Type": "Reference Baseline",
        "Model": "Persistence Baseline",
        "Modality": "Numerical (Lagged PM t)",
        "Target_Task": "Next-hour (t+1) Forecasting",
        "Target_Pollutant": "PM10",
        "Test_Samples": 404431,
        "Independent_Test_Groups": "Held-out temporal split",
        "MAE": 19.7384,
        "RMSE": 36.2374,
        "R2": 0.9115,
        "Evaluation_Notes": "Standard persistence benchmark."
    },
    {
        "Experiment_Type": "Image-Only Baseline",
        "Model": "ResNet-50 Transfer Learning",
        "Modality": "Image-only (PM25Vision)",
        "Target_Task": "Single-image PM2.5 Estimation",
        "Target_Pollutant": "PM2.5",
        "Test_Samples": 79,
        "Independent_Test_Groups": "18 station-date groups",
        "MAE": 66.2148,
        "RMSE": 80.6779,
        "R2": 0.1524,
        "Evaluation_Notes": "Trained on 439 images, evaluated on 79 held-out group images."
    },
    {
        "Experiment_Type": "Exploratory Matched Analysis",
        "Model": "Numerical Ridge Regressor",
        "Modality": "Numerical (Daily ERA5-Land Weather)",
        "Target_Task": "Daily Station PM2.5 Estimation",
        "Target_Pollutant": "PM2.5",
        "Test_Samples": len(test_idx),
        "Independent_Test_Groups": f"{len(test_groups)} station-date groups",
        "MAE": m_num_25["MAE"],
        "RMSE": m_num_25["RMSE"],
        "R2": m_num_25["R2"],
        "Evaluation_Notes": "Evaluated on matched station-date test split (N=60)."
    },
    {
        "Experiment_Type": "Exploratory Matched Analysis",
        "Model": "Image Ridge Regressor",
        "Modality": "Image-only (ResNet-50 Features)",
        "Target_Task": "Daily Station PM2.5 Estimation",
        "Target_Pollutant": "PM2.5",
        "Test_Samples": len(test_idx),
        "Independent_Test_Groups": f"{len(test_groups)} station-date groups",
        "MAE": m_img_25["MAE"],
        "RMSE": m_img_25["RMSE"],
        "R2": m_img_25["R2"],
        "Evaluation_Notes": "Evaluated on matched station-date test split (N=60)."
    },
    {
        "Experiment_Type": "Exploratory Matched Analysis",
        "Model": "Multimodal Early Fusion (Ridge)",
        "Modality": "Multimodal (ResNet-50 + Weather)",
        "Target_Task": "Daily Station PM2.5 Estimation",
        "Target_Pollutant": "PM2.5",
        "Test_Samples": len(test_idx),
        "Independent_Test_Groups": f"{len(test_groups)} station-date groups",
        "MAE": m_fus_25["MAE"],
        "RMSE": m_fus_25["RMSE"],
        "R2": m_fus_25["R2"],
        "Evaluation_Notes": "Evaluated on matched station-date test split (N=60)."
    },
    {
        "Experiment_Type": "Exploratory Matched Analysis",
        "Model": "Numerical Ridge Regressor",
        "Modality": "Numerical (Daily ERA5-Land Weather)",
        "Target_Task": "Daily Station PM10 Estimation",
        "Target_Pollutant": "PM10",
        "Test_Samples": int(v_te10.sum()),
        "Independent_Test_Groups": f"{len(test_groups)} station-date groups",
        "MAE": m_num_10["MAE"],
        "RMSE": m_num_10["RMSE"],
        "R2": m_num_10["R2"],
        "Evaluation_Notes": "Evaluated on matched station-date test split (N=60)."
    },
    {
        "Experiment_Type": "Exploratory Matched Analysis",
        "Model": "Image Ridge Regressor",
        "Modality": "Image-only (ResNet-50 Features)",
        "Target_Task": "Daily Station PM10 Estimation",
        "Target_Pollutant": "PM10",
        "Test_Samples": int(v_te10.sum()),
        "Independent_Test_Groups": f"{len(test_groups)} station-date groups",
        "MAE": m_img_10["MAE"],
        "RMSE": m_img_10["RMSE"],
        "R2": m_img_10["R2"],
        "Evaluation_Notes": "Evaluated on matched station-date test split (N=60)."
    },
    {
        "Experiment_Type": "Exploratory Matched Analysis",
        "Model": "Multimodal Early Fusion (Ridge)",
        "Modality": "Multimodal (ResNet-50 + Weather)",
        "Target_Task": "Daily Station PM10 Estimation",
        "Target_Pollutant": "PM10",
        "Test_Samples": int(v_te10.sum()),
        "Independent_Test_Groups": f"{len(test_groups)} station-date groups",
        "MAE": m_fus_10["MAE"],
        "RMSE": m_fus_10["RMSE"],
        "R2": m_fus_10["R2"],
        "Evaluation_Notes": "Evaluated on matched station-date test split (N=60)."
    }
]

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(OUT_DIR / "experiment_summary.csv", index=False)
print("Saved experiment_summary.csv")

# ---------------------------------------------------------
# 5. GENERATE PLOTS
# ---------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Plot 1: Prediction vs Ground Truth on Matched Test Set
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

models_data = [
    ("Numerical-Only (Weather)", pred_num_pm25, m_num_25, "#2b5c8f"),
    ("Image-Only (ResNet-50)", pred_img_pm25, m_img_25, "#2a9d8f"),
    ("Multimodal Fusion (Image+Weather)", pred_fus_pm25, m_fus_25, "#e76f51"),
]

min_val = min(y_te_pm25.min(), min(p.min() for _, p, _, _ in models_data)) - 10
max_val = max(y_te_pm25.max(), max(p.max() for _, p, _, _ in models_data)) + 10

for ax, (name, preds, m, color) in zip(axes, models_data):
    ax.scatter(y_te_pm25, preds, alpha=0.7, color=color, edgecolors="k", s=45)
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="Ideal 1:1 line")
    ax.set_title(f"{name}\nMAE: {m['MAE']:.1f} | RMSE: {m['RMSE']:.1f} | R²: {m['R2']:.2f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Ground Truth Station PM2.5 (µg/m³)", fontsize=10)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, linestyle=":", alpha=0.6)
    if ax == axes[0]:
        ax.set_ylabel("Predicted PM2.5 (µg/m³)", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)

plt.suptitle("Controlled Matched Test Set: Predictions vs Ground Truth (N=60, 9 Groups)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT_DIR / "matched_predictions_vs_ground_truth.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved matched_predictions_vs_ground_truth.png")

# Plot 2: Alignment Discrepancy & Spatial Distance Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Distance histogram
ax1.hist(distances, bins=35, color="#457b9d", edgecolor="white", alpha=0.85)
ax1.axvline(5.0, color="#e63946", linestyle="--", lw=2, label="5 km Threshold (316 in, 324 out)")
ax1.set_title("PM25Vision Distance to Nearest CPCB Station", fontsize=11, fontweight="bold")
ax1.set_xlabel("Haversine Distance (km)", fontsize=10)
ax1.set_ylabel("Number of Images", fontsize=10)
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, linestyle=":", alpha=0.6)

# Discrepancy scatter
ax2.scatter(matched["PM2.5_mean"], matched["pm25_label"], color="#e76f51", alpha=0.6, edgecolors="none", s=40)
min_p = min(matched["PM2.5_mean"].min(), matched["pm25_label"].min())
max_p = max(matched["PM2.5_mean"].max(), matched["pm25_label"].max())
ax2.plot([min_p, max_p], [min_p, max_p], "k--", lw=1.5, label="1:1 Station = Image")
ax2.plot([min_p, max_p], [min_p + 70.74, max_p + 70.74], "r:", lw=1.5, label="Mean Bias (+70.7 µg/m³)")
ax2.set_title("Label Discrepancy: Image PM2.5 vs Station Daily Ground Truth", fontsize=11, fontweight="bold")
ax2.set_xlabel("Station Daily Mean PM2.5 (µg/m³)", fontsize=10)
ax2.set_ylabel("PM25Vision Image PM2.5 Label (µg/m³)", fontsize=10)
ax2.legend(loc="lower right", fontsize=9)
ax2.grid(True, linestyle=":", alpha=0.6)

plt.suptitle("PM25Vision Alignment Diagnostics (India Urban Subset)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT_DIR / "alignment_discrepancy_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved alignment_discrepancy_distribution.png")

# Plot 3: Residual Distribution on Matched Test Set
fig, ax = plt.subplots(figsize=(8, 5))
res_num = pred_num_pm25 - y_te_pm25
res_img = pred_img_pm25 - y_te_pm25
res_fus = pred_fus_pm25 - y_te_pm25

ax.hist(res_img, bins=15, alpha=0.6, color="#2a9d8f", label=f"Image-Only (MAE={m_img_25['MAE']:.1f})")
ax.hist(res_fus, bins=15, alpha=0.6, color="#e76f51", label=f"Multimodal Fusion (MAE={m_fus_25['MAE']:.1f})")
ax.axvline(0, color="black", linestyle="--", lw=1.5)
ax.set_title("Error Residuals on Held-Out Matched Test Groups (N=60)", fontsize=11, fontweight="bold")
ax.set_xlabel("Residual: Predicted - Actual Ground Truth (µg/m³)", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
fig.savefig(OUT_DIR / "matched_residual_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved matched_residual_distribution.png")

print("\nAll multimodal feasibility and experiment outputs successfully generated.")
