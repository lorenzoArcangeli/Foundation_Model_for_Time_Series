# SKIPP'D Dataset Preparation & Cleaning Logic

This document details the preprocessing, cleaning, and imputation strategy applied to the SKIPP'D dataset to prepare it for Chronos-2 training.

## 1. Objectives
- **Regularity:** Convert irregular timestamps to a strict **30-minute** grid.
- **Continuity:** Fill small, transient gaps caused by sensor dropouts or sunset issues.
- **Authenticity:** Preserve large, genuine outages (e.g., sensor failures lasting weeks) to avoid training on synthetic hallucinations.
- **Completeness:** Ensure the dataset starts strictly at `2017-05-08 07:30:00` and handles image data consistently.

## 2. Pipeline Overview

### Step 1: Truncation & Filtering
- **Cutoff:** Data after `2018-11-03 18:37` is dropped.
- **Start Date:** Data before `2017-05-08 07:30:00-07:00` is dropped.
    - *Note:* The raw data had a large gap immediately following this start date. This gap was explicitly restored (re-indexed) to allows the imputation logic to fill it.

### Step 2: Resampling
- Timestamps are rounded to the nearest 30 minutes.
- Duplicates are dropped (keeping the first occurrence).
- The data is resampled to a strict 30-minute frequency, introducing `NaN` rows for any missing time steps.

### Step 3: Gap Protection
- **Rule:** Any gap longer than **7 days** is marked as "Protected".
- **Reasoning:** These are considered genuine equipment failures. We do not attempt to fill them (except for potential Yearly fallback checks) to ensure the model learns to handle real-world disconnects.
- **Result:** Only 2 massive gaps remain in the final dataset (Nov-Dec 2017, Mar-May 2018).

### Step 4: Multi-Tiered Imputation Strategy
We apply a hierarchical strategy to fill gaps, prioritizing high-confidence short-term fixes before attempting long-term historical patches.

#### Tier 0: Night-Time Fill
- **Logic:** ROI is solar production. Night hours (`20:00` to `08:00`) are strictly zero.
- **Action:**
    - `pv`: Set to `0.0`.
    - `image`: Set to a generated **64x64 Black Image**.

#### Tier 1: Extended Forward Fill
- **Target:** Gaps $\le$ 4 hours.
- **Use Case:** Fixes "Winter Sunset" dropouts where sensors cut off early, and minor transitory loss.
- **Action:** Forward fill values and images from the last valid step.

#### Tier 2: Seasonality (Daily & Weekly)
- **Target:** Gaps < 8 days.
- **Logic:** Solar patterns are highly repetitive.
- **Recursive Backfill:** Checks `t - 24h` (yesterday). If missing, recursively checks up to 7 days back.
- **Weekly Bidirectional:** If daily fails, checks `t - 7d` (last week) or `t + 7d` (next week).
- **Action:** Copy `pv` and `image` from the source timestamp.

#### Tier 3: Yearly Fallback
- **Target:** Stubborn gaps where daily/weekly history is missing (e.g., the very start of the dataset).
- **Logic:** Checks `t + 1 year` (or `t - 1 year`).
- **Action:** Copy `pv` and `image` from the same date in the adjacent year.
- **Key Success:** This logic successfully filled the restorative gap at the start (`2017-05-08` to `2017-05-12`) using data from 2018.

### Step 5: Ad-Hoc Final Polish
Specific manual overrides requested to fix stubborn edge cases:
1.  **2017-05-10 (11:00 - 20:00):** Forced to `0.0` / Black Image.
2.  **2017-03-09 (18:00 - 20:00):** Forced to `0.0` / Black Image (Note: Outside valid range, effectively ignored).
3.  **2017-03-10 (07:30 - 09:30):** Forward Filled (Note: Outside valid range, effectively ignored).

## 3. Image Handling
- **Raw Images:** Preserved where available.
- **Black Images:** Generated `64x64 RGB (0,0,0)` JPEGs used for Night Time fills and zero-value Ad-Hoc fixes.
- **Copied Images:** When `pv` is filled via Forward/Recursive/Yearly logic, the corresponding *source* image is also copied to maintain consistency between visual and numeric data.

## 4. Final Outputs (v12)

### 1. `skippd_train_cleaned_30min_with_images_v12.parquet`
- **Content:** Time, PV, Images, Series ID.
- **Use Case:** Main dataset for Chronos-2 (Visual + Time Series).

### 2. `skippd_train_cleaned_30min_no_images_v12.parquet`
- **Content:** Time, PV, Series ID.
- **Use Case:** Lightweight version for pure time-series baselines or analysis.
