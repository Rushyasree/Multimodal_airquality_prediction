# Multimodal Air Quality Feasibility Audit & Experiment Summary

## 1. Executive Summary & Scientific Verdict

This directory documents the rigorous feasibility audit and controlled experimental evaluation of multimodal fusion for air quality prediction across Indian urban environments.

### Core Scientific Verdict
> **Next-hour continuous multimodal forecasting (ResNet-50 + 24-hour LSTM) is scientifically and statistically indefensible given the current dataset properties.**
>
> 1. **Temporal Mismatch**: The numerical forecasting task requires consecutive hourly sequence intervals ($t-23, \dots, t \rightarrow t+1$). The PM25Vision dataset provides opportunistic outdoor photos with only date-level timestamps (`YYYY-MM-DD`) and no hour/minute metadata. Because intraday PM2.5 fluctuates by an average of $166.72\ \mu\text{g/m}^3$ within a single day at matched monitors, assigning an image to an arbitrary sequence hour introduces severe observational noise.
> 2. **Effective Sample Size Bottleneck**: Across 5.5 years (2015–2020) and 90 CPCB ground stations, only **265 images** fall within 5.0 km of an active monitor on the same calendar date. These 265 images represent only **36 independent station-date groups** across only 19 stations.
> 3. **Severe Burst Skew**: Two single-day bursts account for 40% of all matched images (63 images at DL002 on 2016-01-30 and 42 images at DL026 on 2018-10-24). Group-aware evaluation on 36 independent events yields approximately 7 to 9 test groups—far too small for training or validating complex multi-branch deep architectures without severe overfitting or spurious correlation.
> 4. **Target Discrepancy & Systematic Bias**: PM25Vision labels exhibit an average positive bias of $+70.74\ \mu\text{g/m}^3$ over ground station daily averages (mean $236.83$ vs $166.09\ \mu\text{g/m}^3$).

Consequently, the scientifically defensible path for the research paper is to:
- Preserve the **Numerical LSTM** as the primary high-confidence next-hour forecasting system ($N=404,431$ test sequences, $\text{PM}_{2.5}\ R^2 = 0.9492$).
- Present the **ResNet-50 CNN** as an independent image-only exploratory benchmark ($N=640$ images, $\text{PM}_{2.5}\ R^2 = 0.1524$).
- Report the controlled **Matched Experiment ($N=265$, 36 groups)** as a negative result / methodological boundary condition showing why opportunistic imagery fails to enhance air quality forecasting.

---

## 2. Answers to Phase 2 Feasibility Audit Questions

| Audit Question | Exact Empirical Finding | Scientific Implication |
| :--- | :--- | :--- |
| **Q1. Available IDs, contents, & groups?** | 640 unique image IDs, 640 unique SHA-256 bit-level hashes, 90 station-date groups across India, 35 nearest stations. | Image dataset is clean and deduplicated, but geographically sparse. |
| **Q2. Spatial & temporal matching tolerance?** | 316 images $\le 5.0$ km (min 0.05 km, median 5.16 km, max 620.8 km). Manifest dates cover 2015-09-06 to 2020-02-29, but lack hour timestamps. | Over 50% of images (324) are $>5$ km from any CPCB monitor and cannot be paired with station sensors. |
| **Q3. Independent matched groups?** | 265 images share a date with an active station $\le 5.0$ km, representing only **36 independent station-date groups** across 19 stations. | Effective statistical sample size for multimodal alignment is $N_{\text{groups}} = 36$, not 265. |
| **Q4. Nature of repeated images?** | Distinct camera snapshots (distinct SHA-256) taken in rapid sequence at nearly identical GPS coordinates, assigned identical static PM2.5 labels (e.g. DL002: 63 images all labeled $394.0\ \mu\text{g/m}^3$). | High sample counts mask extreme group pseudo-replication. Random splits would cause severe test set leakage. |
| **Q5. Supported forecasting tasks?** | Intraday station PM2.5 swings by an average of $166.72\ \mu\text{g/m}^3$ (daily min $85.2$, max $251.9\ \mu\text{g/m}^3$). | Next-hour forecasting is invalid without capture time. Only same-day exploratory estimation is possible. |
| **Q6. Statistical power for fusion?** | A 25% test split provides only **9 independent groups** (60 images). | Inadequate statistical power for training multi-parameter deep fusion heads. High variance across test folds. |

---

## 3. Experimental Results Summary

All models were evaluated using strict leakage-free splits (`GroupShuffleSplit` on `station_date_group`).

```
========================================================================================================================
MODEL COMPARISON MATRIX
========================================================================================================================
Experiment Type           Model                   Modality                 Target      Test N   Test Groups   MAE     RMSE     R²
------------------------------------------------------------------------------------------------------------------------
Primary Forecasting       2-Layer LSTM            Numerical (AQ + Weather) PM2.5 (t+1) 404,431  Temporal     10.57   18.58  0.9492
Primary Forecasting       2-Layer LSTM            Numerical (AQ + Weather) PM10  (t+1) 404,431  Temporal     18.86   32.20  0.9301
Reference Benchmark       Persistence             Numerical (Lagged PM t)  PM2.5 (t+1) 404,431  Temporal     10.68   19.81  0.9423
Reference Benchmark       Persistence             Numerical (Lagged PM t)  PM10  (t+1) 404,431  Temporal     19.74   36.24  0.9115
------------------------------------------------------------------------------------------------------------------------
Image-Only Baseline       ResNet-50 Transfer      Image-only (PM25Vision)  PM2.5        79      18 Groups    66.21   80.68  0.1524
------------------------------------------------------------------------------------------------------------------------
Exploratory Matched Sub.  Numerical Ridge         Weather-only             Daily PM2.5  60       9 Groups   193.18  219.87 -5.4129
Exploratory Matched Sub.  Image Ridge (ResNet-50) Image-only (2048 feats)  Daily PM2.5  60       9 Groups    73.67   95.38 -0.2070
Exploratory Matched Sub.  Early Fusion (Ridge)    Multimodal (Img + Wx)    Daily PM2.5  60       9 Groups    85.86  106.56 -0.5064
Exploratory Matched Sub.  Numerical Ridge         Weather-only             Daily PM10   60       9 Groups   297.12  349.23 -3.2755
Exploratory Matched Sub.  Image Ridge (ResNet-50) Image-only (2048 feats)  Daily PM10   60       9 Groups   137.65  172.13 -0.0387
Exploratory Matched Sub.  Early Fusion (Ridge)    Multimodal (Img + Wx)    Daily PM10   60       9 Groups   150.47  186.16 -0.2150
========================================================================================================================
```

### Key Experimental Insights
1. **Multimodal Early Fusion fails to improve generalization over Image-Only**:
   On the 9 held-out test groups, adding daily meteorological variables increased PM2.5 MAE from $73.67$ to $85.86\ \mu\text{g/m}^3$ ($R^2$ dropped from $-0.21$ to $-0.51$). With only 27 training groups, the additional degrees of freedom lead to overfitting.
2. **Weather-only model cannot generalize from 27 daily snapshots**:
   $R^2 = -5.4129$ confirms that static daily meteorological averages without historical sequence context or spatial station embeddings cannot infer absolute pollution concentrations across unseen cities.
3. **Negative $R^2$ demonstrates domain shift**:
   Evaluating across unseen station-date groups reveals that optical atmospheric features exhibit substantial location-specific domain shifts across Indian urban centers.

---

## 4. Generated Artifacts & Visualizations

- **[`feasibility_report.json`](feasibility_report.json)**: Machine-readable audit data answering Phase 2 questions with exact sample counts, distributions, and metrics.
- **[`experiment_summary.csv`](experiment_summary.csv)**: Full tabular comparison of primary forecasting baselines, image-only models, and exploratory matched fusion.
- **[`matched_predictions_vs_ground_truth.png`](matched_predictions_vs_ground_truth.png)**: Three-panel scatter plot of predictions versus ground truth for Numerical-Only, Image-Only, and Multimodal Fusion on the matched test set.
- **[`alignment_discrepancy_distribution.png`](alignment_discrepancy_distribution.png)**: Spatial distance histogram and scatter plot showing the $+70.74\ \mu\text{g/m}^3$ systematic label discrepancy between PM25Vision and CPCB stations.
- **[`matched_residual_distribution.png`](matched_residual_distribution.png)**: Prediction residual histogram contrasting Image-Only and Multimodal Fusion.

---

## 5. What Additional Data Would Be Needed for True Multimodal Fusion?

To construct a scientifically credible next-hour multimodal air quality forecasting system, the research community requires:
1. **Sub-Daily Timestamps**: Second- or minute-level EXIF timestamps matching continuous monitoring loggers.
2. **Co-Located Fixed Cameras**: Continuously operating fixed outdoor cameras installed directly at monitoring station sites, eliminating the $<5.0$ km spatial displacement and enabling synchronous time-lapse sequence extraction.
3. **Calibrated Reference Labels**: Consistent radiometric calibration to prevent optical saturation under severe particulate smog.

---

## 6. Reproducibility

To regenerate the feasibility report, summaries, and plots from the processed datasets:

```powershell
# Ensure virtual environment is active
& ".\.venv\Scripts\python.exe" "scripts\generate_multimodal_artifacts.py"
```
