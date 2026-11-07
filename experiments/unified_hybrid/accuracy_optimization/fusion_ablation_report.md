# VayuDrishti: Fusion Modality Ablation & Image Branch Contribution Report

**Artifact Location**: `experiments/unified_hybrid/accuracy_optimization/fusion_ablation_report.md`  
**Evaluation Scope**: Held-Out Test Set ($N = 97$ samples, $K = 9$ station-date groups)  
**Associated Artifacts**:
- `experiments/unified_hybrid/accuracy_optimization/image_branch_ablations.json`
- `experiments/unified_hybrid/accuracy_optimization/gate_residual_diagnostics.json`
- `experiments/unified_hybrid/accuracy_optimization/paired_group_bootstrap.json`

---

## 1. Executive Summary

This study rigorously investigates whether outdoor camera imagery provides an authentic predictive signal for next-hour air quality forecasting when fused with continuous sensor telemetry. Prior work reported a nominal improvement from PM2.5 MAE of **9.0434 µg/m³** (pretrained sensor LSTM) to **8.9798 µg/m³** (optimized residual hybrid).

Through controlled ablation experiments on the **fixed, trained model**, we evaluate five experimental conditions:
1. **Sensor-Only Fallback** (modality mask $m = 0$)
2. **Full Hybrid with Matched Images** (correct camera-station pairs, $m = 1$)
3. **Full Hybrid with Images Masked** (actual features loaded, $m = 0$)
4. **Full Hybrid with Images Shuffled** (camera features randomly permuted across test instances over 50 Monte Carlo trials)
5. **Full Hybrid with Neutral Representations** (zero vector and training-set feature centroid with $m = 1$)

### Key Scientific Findings:
- **Exact Fallback Verification**: Modality masking ($m = 0$) mathematically guarantees identical equivalence to the sensor baseline (MAE = **9.0434 µg/m³**), confirming that missing or corrupt imagery incurs zero performance penalty.
- **Image Shuffling Invariance**: Randomly permuting image features across instances yielded a mean test PM2.5 MAE of **8.9762 ± 0.0030 µg/m³**, which is virtually identical to (and slightly lower than) the matched image MAE (**8.9798 µg/m³**).
- **Neutral Representation Invariance**: Replacing image features with a constant zero vector or the empirical training centroid with active gating ($m = 1$) yielded test MAEs of **8.9569 µg/m³** and **8.9251 µg/m³**, respectively.
- **Core Conclusion**: The minor numerical improvement of the hybrid model on this test cohort is driven by a small, conservative **global calibration bias adjustment** (+0.08 µg/m³) rather than instance-specific spatial-optical haze extraction. Outdoor camera images without sub-daily timestamps act as static optical context rather than dynamic next-hour predictors.

---

## 2. Experimental Ablation Conditions & Results

All conditions were evaluated on the exact same 97 test instances with identical ground-truth targets (`station_target_pm25` and `station_target_pm10`) using the fixed checkpoint `best_optimized_hybrid_model.pt`.

| Condition | Visual Input | Modality Mask ($m$) | PM2.5 MAE (µg/m³) | PM2.5 RMSE (µg/m³) | PM2.5 $R^2$ | PM10 MAE (µg/m³) | Mean Gate $\alpha$ | Mean Residual $\Delta_{2.5}$ (µg/m³) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Sensor Fallback** | Zero Tensor | $0.0$ | 9.0434 | 15.5682 | 0.9625 | 15.8173 | 0.0000 | -0.0118 |
| **B. Matched Images** | Matched ResNet-50 Feats | $1.0$ | 8.9798 | 15.5225 | 0.9628 | 15.8821 | 0.7794 | +0.0840 |
| **C. Masked Images** | Matched ResNet-50 Feats | $0.0$ | 9.0434 | 15.5682 | 0.9625 | 15.8173 | 0.0000 | -0.0118 |
| **D. Shuffled Images** | Permuted Feats (50 seeds) | $1.0$ | 8.9762 | 15.5210 | 0.9628 | 15.8840 | 0.7793 | +0.0881 |
| **E1. Neutral Zero** | All-Zero Vector ($\mathbf{0}$) | $1.0$ | 8.9569 | 15.5183 | 0.9628 | 15.9114 | 0.7395 | +0.1399 |
| **E2. Neutral Mean** | Training Centroid ($\mathbf{\mu}$) | $1.0$ | 8.9251 | 15.5005 | 0.9629 | 15.9219 | 0.8149 | +0.1751 |

---

## 3. Detailed Analysis of Ablation Behaviors

### 3.1 Verification of Fallback Mechanics (Condition A vs C)
- In Condition A, no image features were supplied, triggering default zero-tensors and $m = 0$.
- In Condition C, genuine matched 2048-dim ResNet-50 features were loaded into memory, but $m$ was forced to $0$.
- **Result**: Predictions under both conditions were identical to 8 decimal places ($\text{MAE} = 9.0434$, $\alpha = 0.0000$). This proves the gating mechanism $\alpha_{\text{eff}} = \alpha_{\text{raw}} \cdot m$ has zero numerical leakage.

### 3.2 Feature-Target Decoupling via Shuffling (Condition B vs D)
- In Condition D, the pairing between the 24-hour sensor sequence at station $S_i$ on date $T_i$ and the camera image at location $L_i$ was severed by randomly shuffling image features across all 97 test samples over 50 independent trials.
- **Hypothesis**: If the hybrid model relied on localized visual optical depth (e.g. plume detection, contrast attenuation) to refine sensor forecasts, shuffling images would mismatch local atmospheric haze, degrading predictive performance ($\text{MAE}_{\text{shuffled}} > \text{MAE}_{\text{matched}}$).
- **Empirical Result**: Shuffled MAE averaged **8.9762 µg/m³** (range: [8.9693, 8.9819]), virtually indistinguishable from the matched baseline of **8.9798 µg/m³**.
- **Implication**: The model does not exploit fine-grained, image-specific optical features to adjust next-hour concentrations.

### 3.3 Neutral Representation Response (Condition E1 & E2)
- In Condition E1, an all-zero vector was passed with $m = 1.0$. In Condition E2, the 2048-dimensional mean vector computed across all 81 training samples was passed.
- **Empirical Result**: When provided an uninformative feature vector, the residual head predicted an offset of $\Delta \approx +0.14$ to $+0.18$ µg/m³.
- Because the base sensor LSTM on this specific test cohort has a slight negative bias (mean error: $+0.09$ µg/m³), adding a slight positive constant improves test MAE slightly across all samples. This accounts for the entire observed numerical gap between 9.0434 and 8.9798 µg/m³.

---

## 4. Optical Metric Correlation Analysis

We extracted physical optical metrics from all 97 test photographs using PIL and OpenCV/SciPy:
- **Sharpness**: 2D Laplacian variance on grayscale pixels (range: 35.4 – 4018.8, mean: 968.1)
- **Contrast**: Standard deviation of luma intensity (range: 48.1 – 88.0, mean: 66.7)
- **Brightness**: Mean luma intensity (range: 70.6 – 180.6, mean: 112.9)

### Correlation Findings:
1. **Gate $\alpha$ vs Geodesic Distance ($d$)**:
   - Pearson $r = -0.3212$ ($p = 0.0013$), Spearman $\rho = -0.2875$ ($p = 0.0043$).
   - The adaptive gating network successfully learned spatial attenuation: as distance between camera and station increases from 0.1 to 1.8 km, the visual gate weight systematically decreases.
2. **Gate $\alpha$ vs Atmospheric Sharpness**:
   - Pearson $r = -0.4101$ ($p < 0.0001$).
   - Lower optical sharpness (higher haze scattering) correlates with a higher visual gating weight.
3. **Residual Delta $\Delta$ vs Sensor Error**:
   - Pearson $r = +0.6634$ ($p < 10^{-12}$), Spearman $\rho = +0.7731$.
   - The residual head directionally aligns with the sign of the sensor forecast error, but its bounded magnitude ($\text{std} = 0.0968$ µg/m³) prevents it from over-correcting.

---

## 5. Scientific Limitations of Current Multimodal Pairing

1. **Sub-Daily Timestamp Absence**: Public outdoor camera benchmarks (including PM25Vision) record only the calendar date. In Indian urban areas where diurnal boundary layer dynamics induce PM2.5 swings $> 150$ µg/m³ between noon and evening, static midday photographs cannot resolve next-hour temporal transients.
2. **Modality Dominance**: High-frequency hourly sensor telemetry contains 24 consecutive historical steps across 27 channels, which captures atmospheric momentum with $R^2 > 0.96$. A single photographic snapshot provides static optical depth, which is largely redundant when continuous telemetry is present.
3. **Academic Position**: The hybrid model is scientifically valid, mathematically robust against negative transfer, and safely deploys in production due to verified fallback. However, it should be described as **competitive and resilient**, rather than conclusively superior to pure telemetry, until edge cameras with synchronized sub-daily timestamps are deployed.
