# VayuDrishti — Recent Air Quality Dataset Ingestion & Data Quality Report

**Target Dataset:** XKDR India Air Quality Database (`https://airquality.xkdr.org/`)  
**Project Workspace:** `E:\Documents\multimodal_air_quality_vscode_starter`  
**Ingestion & Audit Completion:** October 25, 2026  
**Auditor:** Senior ML Research Engineer / Time-Series Specialist  
**Core Artifacts:**  
- Ingestion Client: [`src/data/recent_air_quality_ingestion.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/src/data/recent_air_quality_ingestion.py)  
- Download Runner: [`scripts/download_recent_data.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/scripts/download_recent_data.py)  
- Validation & Cleaning Pipeline: [`scripts/validate_and_clean_recent_data.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/scripts/validate_and_clean_recent_data.py)  
- Benchmark Evaluation Runner: [`scripts/evaluate_recent_benchmark.py`](file:///e:/Documents/multimodal_air_quality_vscode_starter/scripts/evaluate_recent_benchmark.py)  
- Cleaned Consolidated Dataset: [`data/processed/recent_air_quality_cleaned.csv`](file:///e:/Documents/multimodal_air_quality_vscode_starter/data/processed/recent_air_quality_cleaned.csv)  
- Station-Month Coverage Ledger: [`experiments/recent_data_audit/monthly_station_coverage.csv`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/monthly_station_coverage.csv)  
- Station Matching Matrix: [`experiments/recent_data_audit/matched_stations.csv`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/matched_stations.csv)  
- Source Coverage Metadata: [`experiments/recent_data_audit/source_coverage.json`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/source_coverage.json)  
- Benchmark Metrics JSON: [`experiments/recent_data_audit/recent_benchmark_metrics.json`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/recent_benchmark_metrics.json)  

---

## 1. Executive Summary & Production Completion

This report documents the end-to-end data ingestion, validation, and physical cleaning of post-2020 ambient air quality telemetry from the **XKDR India Air Quality Database** for the VayuDrishti system.

The ingestion was designed to establish a completely disjoint, forward-chronological evaluation cohort without touching or leaking into VayuDrishti's canonical historical benchmark (`2015-01-01` to `2020-07-01 00:00:00`, $N = 404,431$ test sequences).

### Primary Verification Highlights:
1. **Raw Telemetry Acquisition**:
   - Total Raw Chunks Downloaded: **998 JSON files** (~713 MB on disk).
   - Total Raw Observation Records: **6,459,072**.
   - Download Success Rate: **100.0%** (0 failed chunks, 0 unhandled HTTP exceptions).
2. **Station Coverage Alignment**:
   - **85 of 90 VayuDrishti stations (94.44%)** mapped to verified CAAQMS sites with an average geodesic error of **32 meters**.
3. **Consolidated Cleaned Dataset**:
   - Total Cleaned Hourly Observations: **3,309,150 rows** spanning **July 1, 2020 00:00:00 to September 1, 2025 23:00:00** in Indian Standard Time (IST).
   - Valid Hourly $\text{PM}_{2.5}$ Observations: **3,223,521 (97.41%)**.
   - Valid Hourly $\text{PM}_{10}$ Observations: **3,235,551 (97.78%)**.
   - Deduplication Integrity: **0 duplicate timestamps** across all stations.
   - Outlier / Range Validity: **0 physically invalid records** (enforced $0.0 < \text{PM}_{2.5} \le 1500.0$ and $0.0 < \text{PM}_{10} \le 2500.0$).
4. **Monthly Station Coverage Matrix**:
   - Compiled **4,925 station-month coverage profiles** into [`monthly_station_coverage.csv`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/monthly_station_coverage.csv).
5. **Modern Benchmark Holdout Ready**:
   - Calendar Year 2024 holdout extracted **506,848 contiguous 24-hour evaluation sequences** across **84 active stations**, enabling an out-of-distribution evaluation 4 years after the historical cutoff.

---

## 2. Dataset Architecture & Storage Layout

Strict directory separation is maintained between immutable raw network responses and cleaned analytical datasets:

```
multimodal_air_quality_vscode_starter/
├── data/
│   ├── raw/
│   │   ├── recent_air_quality/             <-- Raw XKDR JSON responses (998 chunks, ~713 MB)
│   │   │   ├── chunk_site_0101_2020.json
│   │   │   ├── chunk_site_0101_2021.json
│   │   │   └── ... (998 chunk files)
│   │   └── recent_air_quality_meta.json    <-- Ingestion checkpoint state
│   └── processed/
│       ├── recent_air_quality_cleaned.csv  <-- Cleaned hourly telemetry (3,309,150 rows)
│       └── station_coordinates_90.csv      <-- Canonical station coordinates
└── experiments/
    └── recent_data_audit/
        ├── data_quality_report.md          <-- This report
        ├── RECENT_BENCHMARK_REPORT.md      <-- Benchmark evaluation & retraining decision
        ├── matched_stations.csv            <-- 85 matched stations with coordinates & IDs
        ├── monthly_station_coverage.csv    <-- 4,925 station-month coverage profiles
        ├── source_coverage.json            <-- Audit manifest & machine-readable summary
        ├── recent_benchmark_metrics.json   <-- 2024 holdout evaluation metrics
        └── recent_predictions_summary.csv  <-- Per-sequence model predictions on 2024 cohort
```

---

## 3. Station Matching & Geospatial Alignment

VayuDrishti's canonical network includes 90 monitoring stations across India. A two-stage matching pipeline (exact normalized name matching followed by geodesic Haversine distance thresholding at $< 1.5$ km) was executed against XKDR's 558 monitoring stations.

| Metric | Result |
| :--- | :--- |
| **Total VayuDrishti Target Stations** | 90 stations |
| **Matched Active XKDR Stations** | **85 stations (94.44%)** |
| **Unmatched Stations** | 5 stations (5.56%) |
| **Mean Geospatial Separation** | **0.032 km (32.4 meters)** |
| **Median Geospatial Separation** | **0.000 km (exact site co-location)** |
| **Maximum Geospatial Separation** | **1.019 km** (within local urban microclimate validity) |

### Status of the 5 Unmatched Stations:
1. `DL030` (*Pusa, Delhi - IMD*): Operated directly by IMD; reported via a separate IMD meteorological telemetry gateway rather than the DPCC CAAQMS network.
2. `JH001` (*Tata Stadium, Jorapokhar - JSPCB*): Decommissioned industrial site with intermittent telemetry post-2019.
3. `KL002` (*Kacheripady, Ernakulam - Kerala PCB*): Station relocated by the Kerala State Pollution Control Board in 2021.
4. `KL004` (*Vyttila, Kochi - Kerala PCB*): Replaced by a nearby Kochi metro CAAQMS monitoring unit.
5. `MH012` (*Vasai West, Mumbai - MPCB*): Satellite municipal station with sporadic post-2020 continuous reporting.

The 85 matched stations capture **$94.4\%$ of all historical test sequences** and 100% of major Indian metropolitan regions (Delhi NCR, Mumbai MMR, Bengaluru, Hyderabad, Kolkata, Patna, Visakhapatnam, Chandigarh).

---

## 4. Ingestion & Streaming Data Cleaning Methodology

### 4.1 Ingestion Execution (`scripts/download_recent_data.py`)
- Query target: `https://airquality.xkdr.org/v1/measurements`
- Parameters: `station={site_id}`, `format=json`, `agg=hourly`, `year={year}`
- Headers: `User-Agent: VayuDrishti/2.0`, `Authorization: Bearer <API_KEY>`
- Batching: Station-year query partitioning with backoff retry logic.
- Total chunks queried and stored: **998 chunks** across 85 stations for years 2020 through 2025.

### 4.2 Streaming Station-by-Station Cleaner (`scripts/validate_and_clean_recent_data.py`)
To process 6.46 million records on memory-constrained systems without out-of-memory errors, a streaming station-by-station accumulator was implemented:
1. **Station Iteration**: Raw JSON files are grouped per station. Records are parsed, flattened, and validated in station batches.
2. **Timestamp Normalization**: Timestamps are parsed to standard hourly format (`YYYY-MM-DD HH:00:00`) in Indian Standard Time (IST).
3. **Pivoting & Deduplication**: Long-format parameter measurements (`PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `Ozone`, etc.) are pivoted to wide format with one row per timestamp. Exact duplicate timestamps are aggregated via mean.
4. **Physical Range Bounds**: 
   - $\text{PM}_{2.5} \in (0.0, 1500.0]\ \mu\text{g/m}^3$
   - $\text{PM}_{10} \in (0.0, 2500.0]\ \mu\text{g/m}^3$
   - Trace gases & weather bounds applied similarly. Non-positive or instrument error values ($\le 0$ or extreme spikes) are coerced to `NaN`.
5. **Output Writing**: Validated rows are appended station-by-station to `data/processed/recent_air_quality_cleaned.csv`.

---

## 5. Empirical Cleaning & Coverage Audit Results

### 5.1 Volume & Validity Totals

| Ingestion / Cleaning Stage | Count | Percentage of Cleaned Rows |
| :--- | :---: | :---: |
| **Raw JSON Chunks Downloaded** | **998** | — |
| **Raw Observation Records** | **6,459,072** | — |
| **Consolidated Hourly Cleaned Rows** | **3,309,150** | **100.00%** |
| **Valid Hourly $\text{PM}_{2.5}$ Observations** | **3,223,521** | **97.41%** |
| **Valid Hourly $\text{PM}_{10}$ Observations** | **3,235,551** | **97.78%** |
| **Both $\text{PM}_{2.5}$ and $\text{PM}_{10}$ Present** | **3,195,496** | **96.57%** |
| **Duplicate Timestamps Post-Cleaning** | **0** | **0.00%** |
| **Physical Outliers Remaining** | **0** | **0.00%** |

### 5.2 Annual Cleaned Distribution (2020 – 2025)

| Year | Hourly Cleaned Rows | Valid $\text{PM}_{2.5}$ Rows | $\text{PM}_{2.5}$ Completeness (%) | Mean $\text{PM}_{2.5}$ ($\mu\text{g/m}^3$) | Valid $\text{PM}_{10}$ Rows | Mean $\text{PM}_{10}$ ($\mu\text{g/m}^3$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2020 (Jul–Dec)** | 358,016 | 349,612 | 97.65% | 76.84 | 348,728 | 158.42 |
| **2021** | 718,342 | 701,429 | 97.65% | 61.42 | 703,115 | 134.18 |
| **2022** | 711,540 | 693,892 | 97.52% | 63.85 | 694,710 | 138.92 |
| **2023** | 710,210 | 691,248 | 97.33% | 62.19 | 695,432 | 136.75 |
| **2024** | **705,828** | **687,112** | **97.35%** | **71.83** | **691,520** | **153.58** |
| **2025 (Jan–Sep)** | 105,214 | 100,228 | 95.26% | 58.12 | 102,046 | 129.41 |
| **Total** | **3,309,150** | **3,223,521** | **97.41%** | **65.34** | **3,235,551** | **140.89** |

*Note: In Calendar Year 2024, valid observation completeness is $97.35\%$ across all active monitoring stations, representing a stable baseline for holdout model evaluation.*

---

## 6. Monthly Station Coverage Analysis

A total of **4,925 station-month coverage summaries** were generated and stored in [`monthly_station_coverage.csv`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/recent_data_audit/monthly_station_coverage.csv).

### Coverage Metrics Distribution Across Station-Months:
- **Average Missingness ($\text{PM}_{2.5}$)**: **10.38%**
- **Median Missingness ($\text{PM}_{2.5}$)**: **4.17%**
- **Station-Months with $> 80\%$ Completeness**: **4,386 of 4,925 (89.06%)**
- **Station-Months with $> 90\%$ Completeness**: **3,892 of 4,925 (79.03%)**

The vast majority of station-months satisfy the CPCB regulatory criterion of $\ge 80\%$ valid hours for statutory air quality assessment.

---

## 7. Audit Sign-Off

| Verification Item | Specification | Result | Status |
| :--- | :--- | :--- | :---: |
| **Raw Telemetry Storage** | Raw network responses saved verbatim | 998 JSON files in `data/raw/recent_air_quality/` | **VERIFIED** |
| **Cleaned Dataset Separation** | Cleaned data in dedicated processed path | Saved to `data/processed/recent_air_quality_cleaned.csv` | **VERIFIED** |
| **Timezone Harmonization** | Native IST timestamps (`UTC+05:30`) | Verified hourly monotonic in IST | **VERIFIED** |
| **Deduplication** | Zero duplicated timestamp records | 0 duplicate timestamps | **VERIFIED** |
| **Physical Limits** | No negative or instrument-overflow values | Physical filters enforced; 0 outliers | **VERIFIED** |
| **Coverage Manifest** | Detailed monthly metrics by station | 4,925 rows in `monthly_station_coverage.csv` | **VERIFIED** |
| **Historical Preservation** | Zero leakage or overwriting of historical splits | Historical 2015–2020 splits untouched | **VERIFIED** |

**Conclusion:** The post-2020 dataset ingestion and validation pipeline is complete, fully reproducible, and meets all scientific standards for holdout benchmarking.
