# VayuDrishti — Model Evaluation Summary

## 1. Overview and Scope
This document consolidates offline benchmark evaluation metrics for the **VayuDrishti** 2-layer numerical LSTM sequence forecasting model. All metrics reported herein are derived directly from saved evaluation artifacts:
- Targets: `data/processed/model_sequences/y_test.npy` (404,431 sequences, 2 targets: PM2.5, PM10)
- Predictions: `models/lstm_baseline/test_predictions.npy` (404,431 sequences)
- Continuous Metrics: `models/lstm_baseline/test_metrics.json`
- Baseline Comparison: `models/lstm_baseline/persistence_comparison.json`
- Categorical Evaluation: `reports/classification_metrics_report.json`

## 2. Continuous Regression Performance (Held-Out Test Set)
Evaluated across **404,431 unseen hourly sequences** across 90 national CAAQMS stations:

| Target Pollutant | Unit | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Coefficient of Determination ($R^2$) | Persistence Baseline MAE | Persistence Baseline RMSE | Persistence Baseline $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PM2.5** | µg/m³ | **10.5719** | **18.5789** | **0.9492** | 10.6783 | 19.8072 | 0.9423 |
| **PM10** | µg/m³ | **18.8643** | **32.2027** | **0.9301** | 19.7384 | 36.2374 | 0.9115 |

> **Methodological Note on Regression vs. Accuracy**:
> The coefficient of determination ($R^2$) measures the fraction of target variance explained by the model relative to a mean baseline. It is a continuous dispersion metric and must **never** be interpreted or labeled as "classification accuracy" or "percentage of predictions that are correct".

## 3. Exploratory Concentration-Category Classification (CPCB NAQI Breakpoints)
To provide an interpretable operational perspective for public health warnings, continuous predictions and observations were translated into official Central Pollution Control Board (CPCB) NAQI concentration category bins:
- **Good**: PM2.5 $\le 30.0$, PM10 $\le 50.0$ µg/m³
- **Satisfactory**: $30.0 <$ PM2.5 $\le 60.0$, $50.0 <$ PM10 $\le 100.0$ µg/m³
- **Moderate**: $60.0 <$ PM2.5 $\le 90.0$, $100.0 <$ PM10 $\le 250.0$ µg/m³
- **Poor**: $90.0 <$ PM2.5 $\le 120.0$, $250.0 <$ PM10 $\le 350.0$ µg/m³
- **Very Poor**: $120.0 <$ PM2.5 $\le 250.0$, $350.0 <$ PM10 $\le 430.0$ µg/m³
- **Severe**: PM2.5 $> 250.0$, PM10 $> 430.0$ µg/m³

### Summary Metrics ($N = 404,431$ sequences; 0 excluded)
- **PM2.5 Exact Category Accuracy**: **77.04%** (311,558 / 404,431)
- **PM10 Exact Category Accuracy**: **80.88%** (327,087 / 404,431)
- **PM2.5 Within $\pm 1$ Adjacent Category Accuracy**: **99.35%** (401,802 / 404,431)
- **PM10 Within $\pm 1$ Adjacent Category Accuracy**: **99.48%** (402,326 / 404,431)
- **PM2.5 Macro-Averaged F1-Score**: **0.7657** (Precision: 0.7710, Recall: 0.7649)
- **PM10 Macro-Averaged F1-Score**: **0.7504** (Precision: 0.7700, Recall: 0.7374)
- **PM2.5 Weighted-Averaged F1-Score**: **0.7731** (Precision: 0.7829, Recall: 0.7704)
- **PM10 Weighted-Averaged F1-Score**: **0.8075** (Precision: 0.8133, Recall: 0.8088)

### Per-Class Details
#### PM2.5
- **Good** (Support: 119,121): Precision = 0.9345, Recall = 0.7681, F1 = 0.8432
- **Satisfactory** (Support: 112,252): Precision = 0.7127, Recall = 0.8053, F1 = 0.7562
- **Moderate** (Support: 65,915): Precision = 0.6594, Recall = 0.7371, F1 = 0.6961
- **Poor** (Support: 35,395): Precision = 0.5912, Recall = 0.5764, F1 = 0.5837
- **Very Poor** (Support: 52,865): Precision = 0.8331, Recall = 0.8396, F1 = 0.8363
- **Severe** (Support: 18,883): Precision = 0.8952, Recall = 0.8627, F1 = 0.8786

#### PM10
- **Good** (Support: 67,725): Precision = 0.9299, Recall = 0.6974, F1 = 0.7970
- **Satisfactory** (Support: 115,718): Precision = 0.7676, Recall = 0.7872, F1 = 0.7773
- **Moderate** (Support: 162,943): Precision = 0.8392, Recall = 0.9138, F1 = 0.8749
- **Poor** (Support: 30,413): Precision = 0.6666, Recall = 0.6884, F1 = 0.6773
- **Very Poor** (Support: 12,097): Precision = 0.5548, Recall = 0.5362, F1 = 0.5454
- **Severe** (Support: 15,535): Precision = 0.8620, Recall = 0.8014, F1 = 0.8306

## 4. Methodological Limitations & Jurisdictional Boundaries
1. **Averaging Period Mismatch**: Official CPCB NAQI breakpoints were formulated based on epidemiological studies utilizing **24-hour truncated averages**. Applying these category thresholds to **1-hour sequential predictions** provides an informative diagnostic categorization, but does not constitute an official 24-hour regulatory compliance metric.
2. **Sub-Index vs. Composite AQI**: True official CPCB AQI is a multi-pollutant composite metric requiring at least 3 active criteria pollutants including particulate matter. Evaluating PM2.5 or PM10 against these category tiers reflects single-pollutant sub-index accuracy, not composite NAQI accuracy.
3. **Boundary Transience**: Misclassifications predominantly occur at sharp category boundaries (e.g., 29 µg/m³ predicted as 31 µg/m³). When evaluated with $\pm 1$ category tolerance, accuracy exceeds 99.3%, reflecting that large ordinal mistakes are extremely rare.
