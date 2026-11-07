# VayuDrishti Phase 13A: Spatial-Temporal Cross-Station Advection & Multi-Node Modeling Report

**Document Version:** 1.0  
**Phase:** Phase 13A — Spatial-Temporal Cross-Station Advection  
**Date:** November 2026  
**Lead ML Auditor:** Spatial Data Science & Atmospheric Modeling Group  
**Scope:** Intra-urban air quality correlation, along-transect wind velocity projection, and multi-node predictive feature utility.

---

## 1. Executive Summary

Phase 13A investigates whether spatial correlation and atmospheric advection across neighboring continuous monitoring stations can improve next-hour ($t+1$) air quality forecasting over single-station autoregression without temporal lookahead leakage. 

Utilizing the expanded 15-station network established in Phase 12, three intra-urban station clusters situated within $25\text{ km}$ of one another were analyzed:
1. **Delhi NCR Urban Corridor:** Anand Vihar (`DL002`) $\leftrightarrow$ DTU (`DL008`) ($d = 22.97\text{ km}$, forward azimuth $\theta = 299.8^\circ$).
2. **Mumbai Coastal Marine Corridor:** Bandra (`MH005`) $\leftrightarrow$ Colaba (`MH008`) ($d = 17.20\text{ km}$, forward azimuth $\theta = 189.2^\circ$).
3. **Kolkata Gangetic Urban Corridor:** Bidhannagar (`WB008`) $\leftrightarrow$ Jadavpur (`WB010`) ($d = 10.07\text{ km}$, forward azimuth $\theta = 204.6^\circ$).

### Key Empirical Findings:
1. **High Synchronous Regional Coupling:**
   - In Kolkata ($10.1\text{ km}$), synchronous correlation between stations is exceptionally high ($r = 0.9078$ for $\text{PM}_{2.5}$, $r = 0.9168$ for $\text{PM}_{10}$), reflecting shared regional airshed dynamics under low advective dispersion.
   - In Delhi ($23.0\text{ km}$), synchronous correlation is strong ($r = 0.7107$ for $\text{PM}_{2.5}$, $r = 0.6669$ for $\text{PM}_{10}$), with peak correlation occurring at $\text{Lag} = 0\text{ hours}$.
   - In Mumbai ($17.2\text{ km}$), peak correlation is observed with a **1-hour lag** ($r = 0.5385$ at $\tau = +1\text{h}$ for $\text{PM}_{2.5}$), indicating coastal sea-breeze along-transect transport.
2. **Causal Predictive Value of Spatial Lags:**
   - In Mumbai, adding neighbor spatial antecedent lags ($\le t$) produces an incremental gain: MAE improves from $6.10$ to $6.04 \ \mu\text{g/m}^3$ ($+0.86\%$ gain over single-station GBDT, $+4.28\%$ over persistence).
   - In Delhi and Kolkata, single-station autoregressive GBDT already captures local diurnal emission cycles effectively. Naively concatenating unweighted spatial lags from distant nodes slightly degrades single-station accuracy ($-2.0\%$ in Delhi, $-3.0\%$ in Kolkata) due to localized microclimate variance and feature dilution.
3. **Core Academic Conclusion:**
   - Unweighted, static spatial lag features are insufficient for landlocked continental megacities where local boundary layer dynamics dominate over regional advection. Spatial multi-node modeling requires explicit, physics-informed wind gating (incorporating along-transect wind velocity $v_{\parallel}$) to selectively activate neighbor information only during verified transport episodes.

---

## 2. Intra-Urban Cluster Geometry & Bearing Analysis

Geodesic distances and forward/backward great-circle azimuths were calculated between all intra-urban station pairs:

| Cluster Name | City | Station A | Station B | Distance $(d)$ | Azimuth $A \to B$ | Azimuth $B \to A$ | Overlapping Hours |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Delhi NCR Corridor** | Delhi | Anand Vihar (`DL002`) | DTU (`DL008`) | $22.97\text{ km}$ | $299.8^\circ$ (NW) | $119.7^\circ$ (SE) | 33,337 |
| **Mumbai Marine Corridor** | Mumbai | Bandra (`MH005`) | Colaba (`MH008`) | $17.20\text{ km}$ | $189.2^\circ$ (S) | $9.2^\circ$ (N) | 17,956 |
| **Kolkata Gangetic Corridor** | Kolkata | Bidhannagar (`WB008`) | Jadavpur (`WB010`) | $10.07\text{ km}$ | $204.6^\circ$ (SSW) | $24.6^\circ$ (NNE) | 40,009 |

---

## 3. Pairwise Cross-Correlation Function (CCF) Analysis

Cross-correlation $r(\tau)$ was evaluated over hourly lags $\tau \in [-12, +12]\text{ hours}$:

$$\text{CCF}(\tau) = \frac{\sum_{t} (x_A(t - \tau) - \bar{x}_A)(x_B(t) - \bar{x}_B)}{\sqrt{\sum_t (x_A(t - \tau) - \bar{x}_A)^2 \sum_t (x_B(t) - \bar{x}_B)^2}}$$

| Cluster | Parameter | Lag 0 Correlation | Peak Lag $\tau^*$ | Peak Correlation $r(\tau^*)$ | Physical Implication |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Delhi Corridor** | $\text{PM}_{2.5}$ | $0.7107$ | $0\text{ h}$ | $0.7107$ | Synchronous diurnal urban emissions dominate. |
| **Delhi Corridor** | $\text{PM}_{10}$ | $0.6669$ | $0\text{ h}$ | $0.6669$ | Synchronous morning/evening traffic peaks. |
| **Mumbai Corridor** | $\text{PM}_{2.5}$ | $0.5344$ | **+1 h** | **0.5385** | Bandra leads Colaba by 1 hour along coastal flow. |
| **Mumbai Corridor** | $\text{PM}_{10}$ | $0.4804$ | **+1 h** | **0.4821** | 1-hour advective lag under sea-breeze circulation. |
| **Kolkata Corridor** | $\text{PM}_{2.5}$ | $0.9078$ | $0\text{ h}$ | $0.9078$ | Tightly coupled deltaic boundary layer. |
| **Kolkata Corridor** | $\text{PM}_{10}$ | $0.9168$ | $0\text{ h}$ | $0.9168$ | Near-perfect spatial synchrony ($10\text{ km}$). |

---

## 4. Along-Transect Wind Velocity & Physical Advection Modeling

Using continuous ERA5-Land reanalysis winds ($u_{10}, v_{10}, v_{\text{wind}}, \phi_{\text{wind}}$), the along-transect wind velocity component was computed:

$$v_{\parallel} = v_{\text{wind}} \cdot \cos(\phi_{\text{wind}} - \theta_{\text{transect}})$$

Physical advection transit time:

$$\tau_{\text{transit}} = \frac{d}{v_{\parallel} \cdot 3.6} \quad (\text{hours})$$

| Cluster Transect | Distance | Transect Azimuth | Favorable Hours ($v_{\parallel} \ge 1.5\text{ m/s}$) | Mean $v_{\parallel}$ | Theoretical Transit Time | Empirical Peak Lag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DL002 $\to$ DL008** | $23.0\text{ km}$ | $299.8^\circ$ | 3,115 hrs | $2.56\text{ m/s}$ | $2.49\text{ hours}$ | $0\text{ hours}$ |
| **WB008 $\to$ WB010** | $10.1\text{ km}$ | $204.6^\circ$ | 4,289 hrs | $3.12\text{ m/s}$ | $0.90\text{ hours}$ | $0\text{ hours}$ |

---

## 5. Causal Spatial-Temporal Predictive Value Evaluation

Evaluated across chronological 80/20 train/test splits (zero lookahead leakage):

| Target Station | Neighbor Station | Test $N$ | Persistence MAE | Single-Station GBDT MAE | Multi-Node Spatial GBDT MAE | Spatial Gain vs. Single (%) | Overall Gain vs. Persistence (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DTU, Delhi (`DL008`)** | Anand Vihar (`DL002`) | 6,667 | $14.74$ | **13.91** | $14.19$ | $-2.03\%$ | **+3.73%** |
| **Colaba, Mumbai (`MH008`)** | Bandra (`MH005`) | 3,591 | $6.31$ | $6.10$ | **6.04** | **+0.86%** | **+4.28%** |
| **Jadavpur, Kolkata (`WB010`)** | Bidhannagar (`WB008`) | 8,001 | $2.94$ | **2.86** | $2.94$ | $-3.00\%$ | **+0.00%** |

### Research Takeaway:
Adding unweighted spatial lags from neighboring stations provides modest utility only in advection-driven coastal environments (Mumbai, $+0.86\%$). In continental basins (Delhi, Kolkata), local single-station autoregression achieves superior accuracy by preserving model capacity for high-frequency in-situ variations.
