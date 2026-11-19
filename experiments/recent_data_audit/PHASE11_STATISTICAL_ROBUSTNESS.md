# VayuDrishti — Phase 11: Statistical Robustness & Sensitivity Analysis

**Audit Date:** November 5, 2026  
**Evaluation Scope:** Complete 14-Day Live Trial Dataset ($N = 418$ Matched Evaluation Pairs)  
**Execution Script:** [`scripts/compute_phase11_robustness.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/scripts/compute_phase11_robustness.py)  
**Quantitative Metrics File:** [`experiments/recent_data_audit/phase11_robustness_metrics.json`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/phase11_robustness_metrics.json)  
**Model Under Evaluation:** Standalone Post-2020 Adapted GBDT (`adapted_gbdt_only`) vs Causal Persistence Baseline  

---

## 1. Executive Summary & Core Statistical Findings

In Phase 11 Task 3, a rigorous statistical robustness and sensitivity analysis was conducted on the full accumulated live dataset from October 22 to November 5, 2026. Prior phases observed that hourly air quality residuals exhibit serial autocorrelation ($\rho_1 \approx 0.38$ for $\text{PM}_{2.5}$, $\rho_1 \approx 0.41$ for $\text{PM}_{10}$), invalidating standard independent and identically distributed (i.i.d.) assumptions and paired Student's $t$-tests.

To address autocorrelation, station heterogeneity, and family-wise error inflation, this evaluation executes:
1. **Station-Stratified Block Bootstrap Sensitivity Analysis** across block lengths $L \in \{24\text{h}, 48\text{h}, 72\text{h}\}$ with $B = 5,000$ iterations.
2. **Multiple-Comparison Corrections** (Bonferroni FWER and Benjamini-Hochberg FDR) across 16 subgroup hypotheses (4 stations $\times$ 2 concentration regimes for both $\text{PM}_{2.5}$ and $\text{PM}_{10}$).
3. **Formal Demarcation** between Pre-specified Confirmatory Hypotheses and Exploratory Subgroup Findings.

### Key Conclusions:
> [!IMPORTANT]
> **Primary Confirmatory Finding: Inconclusive / Not Statistically Significant**  
> On the pooled national dataset ($N=418$), the standalone adapted GBDT achieves a nominal reduction of $\Delta\text{MAE} = -0.223\ \mu\text{g/m}^3$ ($-2.27\%$) for $\text{PM}_{2.5}$. However, under station-stratified block bootstrap resampling, **all 95% confidence intervals cross zero across 24h, 48h, and 72h block lengths** ($p > 0.40$).  
> We **fail to reject the null hypothesis** of zero difference between GBDT and persistence nationally. National superiority cannot be confirmed on 4 stations over 14 days.

> [!NOTE]
> **Exploratory Subgroup Finding: Pronounced Station & Regime Heterogeneity**  
> Subgroup analysis reveals two distinct physical regimes:
> 1. **High-Volatility Urban Episodes (Delhi: `DL002`, `DL008`)**: GBDT significantly outperforms persistence during elevated nocturnal inversions. On `DL008` (Elevated $\text{PM}_{10}$, $n=80$), GBDT cuts MAE by **$6.23\ \mu\text{g/m}^3$** (unadjusted $p = 0.00357$, **Benjamini-Hochberg FDR Significant**).
> 2. **Flat Low-Variance Coastal/Rural Baselines (`AP001`, `AP005`)**: At `AP001` (Amaravati), baseline $\text{PM}_{2.5}$ is ultra-clean ($\approx 2\text{--}4\ \mu\text{g/m}^3$) where persistence achieves $\text{MAE} = 1.45\ \mu\text{g/m}^3$. GBDT exhibits a shrinkage bias toward national medians ($\text{MAE} = 2.09\ \mu\text{g/m}^3$), resulting in persistence outperforming GBDT by $+0.64\ \mu\text{g/m}^3$ (**Bonferroni Significant**, $p < 0.0001$).

---

## 2. Block Bootstrap Sensitivity Analysis

### 2.1 Theoretical Rationale for Block Lengths
Let $\{X_t\}$ be a non-stationary hourly time series of prediction errors. When residuals exhibit autocorrelation, sampling individual hours with replacement breaks temporal dependence, artificially compressing confidence intervals and inflating Type I error rates. The block bootstrap preserves dependence within contiguous time blocks of length $L$.
- **$L = 24\text{ hours}$**: Matches the natural diurnal atmospheric cycle (daytime planetary boundary layer mixing and nocturnal ground-level inversions).
- **$L = 48\text{ hours}$**: Captures multi-day weather persistence and frontal passages.
- **$L = 72\text{ hours}$**: Captures synoptic-scale meteorological regimes, but yields fewer total blocks per station ($14 \times 24 / 72 \approx 4.7$ blocks), increasing block-sampling variance.

### 2.2 Sensitivity Matrix across Block Lengths (Pooled Telemetry, $N=418$)

$$\Delta\text{MAE} = \text{MAE}_{\text{GBDT}} - \text{MAE}_{\text{Persistence}} \quad (\text{Negative values favor GBDT})$$
$$\Delta\text{RMSE} = \text{RMSE}_{\text{GBDT}} - \text{RMSE}_{\text{Persistence}} \quad (\text{Negative values favor GBDT})$$

| Pollutant | Metric | Block Length $L$ | Bootstrap Mean | Bootstrap Std | 95% Bootstrap CI | CI Width | Two-Tailed $p$-value | Significant at $\alpha=0.05$? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\text{PM}_{2.5}$** | **$\Delta\text{MAE}$** | **24h** | $-0.225$ | $0.362$ | **$[-0.920,\ +0.499]$** | $1.419$ | $0.5152$ | **No (Crosses Zero)** |
| **$\text{PM}_{2.5}$** | **$\Delta\text{MAE}$** | **48h** | $-0.239$ | $0.340$ | **$[-0.907,\ +0.419]$** | $1.326$ | $0.4792$ | **No (Crosses Zero)** |
| **$\text{PM}_{2.5}$** | **$\Delta\text{MAE}$** | **72h** | $-0.237$ | $0.297$ | **$[-0.850,\ +0.311]$** | $1.161$ | $0.4096$ | **No (Crosses Zero)** |
| **$\text{PM}_{2.5}$** | **$\Delta\text{RMSE}$** | **24h** | $-0.751$ | $0.646$ | **$[-2.095,\ +0.410]$** | $2.505$ | $0.2408$ | **No (Crosses Zero)** |
| **$\text{PM}_{2.5}$** | **$\Delta\text{RMSE}$** | **48h** | $-0.781$ | $0.627$ | **$[-2.052,\ +0.395]$** | $2.447$ | $0.2184$ | **No (Crosses Zero)** |
| **$\text{PM}_{2.5}$** | **$\Delta\text{RMSE}$** | **72h** | $-0.771$ | $0.583$ | **$[-1.936,\ +0.313]$** | $2.249$ | $0.1872$ | **No (Crosses Zero)** |
| **$\text{PM}_{10}$** | **$\Delta\text{MAE}$** | **24h** | $-0.243$ | $0.875$ | **$[-1.962,\ +1.499]$** | $3.461$ | $0.7712$ | **No (Crosses Zero)** |
| **$\text{PM}_{10}$** | **$\Delta\text{MAE}$** | **48h** | $-0.246$ | $0.814$ | **$[-1.872,\ +1.341]$** | $3.213$ | $0.7584$ | **No (Crosses Zero)** |
| **$\text{PM}_{10}$** | **$\Delta\text{MAE}$** | **72h** | $-0.249$ | $0.697$ | **$[-1.636,\ +1.111]$** | $2.747$ | $0.7208$ | **No (Crosses Zero)** |
| **$\text{PM}_{10}$** | **$\Delta\text{RMSE}$** | **24h** | $-0.781$ | $1.061$ | **$[-2.995,\ +1.148]$** | $4.143$ | $0.4664$ | **No (Crosses Zero)** |
| **$\text{PM}_{10}$** | **$\Delta\text{RMSE}$** | **48h** | $-0.784$ | $0.999$ | **$[-2.810,\ +1.077]$** | $3.887$ | $0.4352$ | **No (Crosses Zero)** |
| **$\text{PM}_{10}$** | **$\Delta\text{RMSE}$** | **72h** | $-0.767$ | $0.887$ | **$[-2.585,\ +0.923]$** | $3.508$ | $0.3888$ | **No (Crosses Zero)** |

### 2.3 Sensitivity Analysis Takeaways
1. **Robustness Across Block Lengths**: The point estimate remains virtually invariant across block lengths ($\Delta\text{MAE}_{\text{PM2.5}} \approx -0.23\ \mu\text{g/m}^3$), confirming that estimation bias is negligible.
2. **Width Behavior**: As block length increases from 24h to 72h, confidence interval bounds remain stable and consistently straddle zero.
3. **Statistical Inconclusiveness**: Under all tested block lengths, the two-tailed empirical $p$-value exceeds $0.40$. The data does **not** support a claim of statistically significant national improvement over persistence.

---

## 3. Station-Stratified Sensitivity & Heterogeneity

Evaluating the 4 stations independently demonstrates the geographic and environmental bifurcation of the model's behavior:

### Station Breakdown at $L = 24\text{h}$ Block Length

| Station ID | Location | Sample Size $n$ | $\text{PM}_{2.5}\ \Delta\text{MAE}$ (Mean) | 95% Block Bootstrap CI | Favored Model |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`AP001`** | Amaravati (Secretariat) | 149 | $+0.650$ | $[+0.238,\ +1.149]$ | **Persistence (Statistically Significant)** |
| **`AP005`** | Visakhapatnam (GVM Corp) | 33 | $+0.584$ | $[-1.026,\ +2.378]$ | Inconclusive (Crosses Zero) |
| **`DL002`** | New Delhi (Anand Vihar) | 131 | $-1.698$ | $[-3.153,\ -0.209]$ | **GBDT (Statistically Significant)** |
| **`DL008`** | New Delhi (DTU) | 105 | $-0.558$ | $[-2.408,\ +1.432]$ | Inconclusive (Crosses Zero) |

```mermaid
flowchart LR
    subgraph South["Southern Flat Baselines (AP001, AP005)"]
        A["Low Variance (PM2.5 ~ 2-4 ug/m3)"] --> B["Persistence is optimal (MAE 1.45)"]
        B --> C["GBDT exhibits +0.64 shrinkage bias"]
    end
    subgraph North["Delhi Urban Corridors (DL002, DL008)"]
        D["High Diurnal Inversions (PM2.5 up to 350 ug/m3)"] --> E["Persistence lags severely during ramps"]
        E --> F["GBDT cuts error by 8.46% (MAE 18.39 vs 20.09)"]
    end
```

---

## 4. Multiple-Comparison Corrections across Subgroups

To prevent False Discovery Rate (FDR) inflation when testing multiple subsets, we evaluated 16 subgroup hypotheses:
$$\text{4 Stations} \times \text{2 Regimes (Standard-Compliant } \le 60\ \mu\text{g/m}^3 \text{ vs Elevated } > 60\ \mu\text{g/m}^3) \times \text{2 Pollutants} = 16\ \text{tests}$$

- **Nominal $\alpha$**: $0.05$
- **Bonferroni Critical Threshold**: $\alpha_{\text{Bonf}} = \frac{0.05}{16} = 0.003125$
- **Benjamini-Hochberg (BH) FDR**: Controls expected proportion of false positives at $q \le 0.05$.

### Complete 16-Hypothesis Multiple-Comparison Table

| Rank | Subgroup Test | Sample Size $n$ | Mean $\Delta\text{MAE}\ (\mu\text{g/m}^3)$ | Unadjusted $p$-value | Bonferroni Adj. $p$ | Bonferroni Significant? | BH Critical Value | BH FDR Significant? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `AP001_Standard_Compliant_PM25` | 149 | $+0.641$ | $< 0.00001$ | $0.00001$ | **YES (Favors Pers)** | $0.00313$ | **YES (Favors Pers)** |
| **2** | `AP001_Standard_Compliant_PM10` | 149 | $+1.212$ | $0.00005$ | $0.00075$ | **YES (Favors Pers)** | $0.00625$ | **YES (Favors Pers)** |
| **3** | `DL008_Standard_Compliant_PM10` | 25 | $+6.720$ | $0.00019$ | $0.00304$ | **YES (Favors Pers)** | $0.00938$ | **YES (Favors Pers)** |
| **4** | `DL008_Elevated_PM10` | 80 | $-6.228$ | **$0.00357$** | $0.05714$ | No (Marginal) | $0.01250$ | **YES (Favors GBDT)** |
| **5** | `DL008_Elevated_PM25` | 80 | $-2.459$ | **$0.01764$** | $0.28217$ | No | $0.01562$ | No |
| **6** | `DL008_Standard_Compliant_PM25` | 25 | $+5.479$ | $0.06363$ | $1.00000$ | No | $0.01875$ | No |
| **7** | `AP005_Elevated_PM25` | 12 | $+1.651$ | $0.06837$ | $1.00000$ | No | $0.02188$ | No |
| **8** | `DL002_Elevated_PM25` | 86 | $-1.268$ | $0.07282$ | $1.00000$ | No | $0.02500$ | No |
| **9** | `DL002_Standard_Compliant_PM25` | 45 | $-0.872$ | $0.11090$ | $1.00000$ | No | $0.02813$ | No |
| **10** | `DL002_Standard_Compliant_PM10` | 45 | $+3.118$ | $0.15197$ | $1.00000$ | No | $0.03125$ | No |
| **11** | `AP005_Standard_Compliant_PM10` | 21 | $-0.522$ | $0.46698$ | $1.00000$ | No | $0.03438$ | No |
| **12** | `DL002_Elevated_PM10` | 86 | $-0.846$ | $0.74091$ | $1.00000$ | No | $0.03750$ | No |
| **13** | `AP005_Standard_Compliant_PM25` | 21 | $-0.030$ | $0.94056$ | $1.00000$ | No | $0.04063$ | No |
| **14** | `AP005_Elevated_PM10` | 12 | $+0.078$ | $0.95632$ | $1.00000$ | No | $0.04375$ | No |
| **15** | `AP001_Elevated_PM25` | 0 | — | $1.00000$ | $1.00000$ | No | $0.04688$ | No |
| **16** | `AP001_Elevated_PM10` | 0 | — | $1.00000$ | $1.00000$ | No | $0.05000$ | No |

---

## 5. Confirmatory vs Exploratory Demarcation

To maintain absolute scientific integrity for future academic publication, results must be explicitly categorized:

### 5.1 Confirmatory Findings (Pre-specified Primary Hypotheses)
1. **Primary National $\text{PM}_{2.5}$ Hypothesis**:
   - *Hypothesis*: Standalone post-2020 adapted GBDT achieves lower $\text{PM}_{2.5}$ MAE than Persistence across all monitored Indian CAAQMS stations.
   - *Result*: **Inconclusive / Not Statistically Significant**.
   - *Empirical Difference*: $\Delta\text{MAE} = -0.223\ \mu\text{g/m}^3$ ($-2.27\%$).
   - *Block Bootstrap CIs*: $L=24\text{h}: [-0.920, 0.499]$; $L=48\text{h}: [-0.907, 0.419]$; $L=72\text{h}: [-0.850, 0.311]$.
   - *Verdict*: Fail to reject null hypothesis.
2. **Primary National $\text{PM}_{10}$ Hypothesis**:
   - *Hypothesis*: Standalone post-2020 adapted GBDT achieves lower $\text{PM}_{10}$ MAE than Persistence nationally.
   - *Result*: **Inconclusive / Not Statistically Significant**.
   - *Empirical Difference*: $\Delta\text{MAE} = -0.221\ \mu\text{g/m}^3$ ($-0.99\%$).
   - *Verdict*: Fail to reject null hypothesis.

### 5.2 Exploratory Findings (Subgroup & Post-Hoc Analysis)
1. **Urban Inversion Dynamics (Delhi Corridors)**:
   - On `DL008` under Elevated $\text{PM}_{10}$ conditions ($n=80$), GBDT reduces MAE by **$6.23\ \mu\text{g/m}^3$** (unadjusted $p = 0.00357$, surviving Benjamini-Hochberg FDR correction at $q \le 0.05$).
   - On `DL002` (Anand Vihar), GBDT achieves an overall $8.46\%$ error reduction ($\Delta\text{MAE} = -1.70\ \mu\text{g/m}^3$), with block bootstrap 95% CI $[-3.15, -0.21]$ excluding zero.
2. **Baseline Persistence Superiority in Clean Environments**:
   - On `AP001` (Amaravati) under standard-compliant conditions ($n=149$), Persistence achieves $\text{MAE} = 1.45\ \mu\text{g/m}^3$, beating GBDT ($2.09\ \mu\text{g/m}^3$). This superiority survives both Bonferroni ($p_{\text{adj}} = 0.00001$) and Benjamini-Hochberg correction.
   - *Physical Mechanism*: When ambient particulate concentrations remain low and stable, statistical regression models introduce minor variance shrinkage bias, whereas identity persistence ($y_{t+1} = y_t$) is statistically optimal.

---

## 6. Scientific Summary & Publication Guidance

The rigorous application of block bootstrap and multiple-comparison adjustments establishes that **the adapted GBDT cannot be claimed as a blanket superior model across India**. Rather, it acts as a specialized non-linear compensator for severe urban pollution ramps. These findings provide a grounded, honest basis for paper presentation without overstated claims.
