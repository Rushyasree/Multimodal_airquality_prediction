# Pairing Audit Report: Sensor–Image Multimodal Alignment

**Project:** VayuDrishti — Urban Air Quality Forecasting  
**Module:** `experiments/multimodal_fusion/`  
**Date:** October 24, 2026  
**Artifact File:** [`experiments/multimodal_fusion/pairing_audit.json`](file:///e:/Documents/multimodal_air_quality_vscode_starter/experiments/multimodal_fusion/pairing_audit.json)

---

## 1. Overview and Scientific Grounding

This pairing audit establishes the exact numerical, spatial, and temporal alignment between the 640 curated PM25Vision outdoor camera images and the 1,067,495 hourly 24-hour sequence records from the 90 Central Pollution Control Board (CPCB) continuous monitoring stations.

### Mandatory Scientific Disclosure:
Strict next-hour causal image fusion **cannot be validated using the current PM25Vision dataset** because image capture hours, minutes, and seconds are entirely absent in both metadata and EXIF image headers. Rather than disguising missing temporal alignment, this working prototype demonstrates the full functioning PyTorch multimodal fusion architecture and its training/inference pipeline on representative midday sensor alignments, while maintaining absolute scientific transparency.

---

## 2. Quantitative Pairing Summary

- **Total Curated Images in Project Manifest**: 640 unique physical images.
- **Images within $\le 5.0\text{ km}$ of a CPCB Monitor**: **316 images** (49.4%).
- **Images Exceeding $5.0\text{ km}$**: **324 images** (50.6%, unmatchable).
- **Unique Station-Date Groups within 5 km**: 48 groups.
- **Usable Paired Images in Sequence Index**: **199 images**.
- **Usable Independent Station-Date Groups**: **33 groups** across **18 monitoring stations**.
- **Distance Statistics for Paired Images**:
  - Minimum distance: $0.052\text{ km}$ ($52\text{ meters}$)
  - Mean distance: $0.850\text{ km}$
  - Median distance: $0.620\text{ km}$
  - Maximum distance: $3.085\text{ km}$

---

## 3. Timestamp Precision and Alignment Policy

| Dimension | Continuous Sensor Telemetry | PM25Vision Camera Images | Alignment Policy Used |
| :--- | :--- | :--- | :--- |
| **Temporal Resolution** | Hourly timestamps (`YYYY-MM-DD HH:00:00`) | Calendar date only (`YYYY-MM-DD`) | Representative Midday Sensor Sequence |
| **Capture Hour Stamp** | Explicit UTC/IST integer hour $[0, 23]$ | **None (0% available in EXIF/tags)** | Closest available target to 12:00 PM noon |
| **Sequence Window** | 24 consecutive hours ($t-23$ to $t$) | Static photographic snapshot | Preceding 24 hours leading up to target |
| **Causal Guarantee** | Confirmed forward-looking ($t \rightarrow t+1$) | **Indeterminate** | Prototype architecture demonstration |

### Diurnal Context:
In urban Indian metropolitan areas, continuous CAAQMS monitors experience an average intraday diurnal swing of **$166.72\ \mu\text{g/m}^3$** between daytime boundary-layer dispersion and nocturnal thermal inversion peaks. Consequently, treating an undated daytime photograph as a verified predictor for an arbitrary nighttime sequence is mathematically non-physical. The representative midday alignment anchors each image to daylight observation conditions.

---

## 4. Target Definition and Systematic Discrepancy

Two competing candidate target definitions exist for paired instances:
1. **Primary Ground-Truth Target (`station_target_pm25`)**:  
   The actual beta-attenuation monitor measurement recorded by the CPCB CAAQMS ground station at the target forecast hour.
   - Mean: **$133.51\ \mu\text{g/m}^3$**
   - Standard Deviation: **$141.36\ \mu\text{g/m}^3$**
   - Range: $[22.65, 500.25]\ \mu\text{g/m}^3$
2. **Web-Scraped Label (`image_label_pm25`)**:  
   The static label associated with the PM25Vision image record.
   - Mean: **$187.85\ \mu\text{g/m}^3$**
   - Standard Deviation: **$64.02\ \mu\text{g/m}^3$**
   - Range: $[98.00, 461.00]\ \mu\text{g/m}^3$
- **Discrepancy Analysis**:  
   The mean absolute discrepancy between image labels and ground-truth station observations is **$96.37\ \mu\text{g/m}^3$**. To maintain scientific integrity, all models in this fusion evaluation predict the physical ground-truth station concentration (`station_target_pm25`).

---

## 5. Burst Clustering & Pseudo-Replication

A major statistical challenge in the paired dataset is burst sampling:
- **Top Group (`DL026__2018-10-24`)**: 42 images (21.1% of paired data).
- **Second Group (`DL031__2016-04-28`)**: 24 images (12.1% of paired data).
- **Top 2 Groups**: Account for **33.2%** of all usable pairs.
- **Top 5 Groups**: Account for **60.3%** of all usable pairs.

**Partitioning Rule**:  
Because images taken on the same date at the same station share identical visual scenes and weather, standard random splitting would produce severe data leakage. A strict `GroupShuffleSplit` on `station_date_group` is mandatory to keep groups 100% intact within splits.
