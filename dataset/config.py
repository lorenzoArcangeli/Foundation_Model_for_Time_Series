import os

"""
Configuration for Dataset Preparation Pipeline.
Centralizes all path definitions, time settings, and imputation thresholds.
"""

# --- File System Paths ---
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SKIPPD_DATA_DIR = os.path.join(_BASE_DIR, "skippd_data")

RAW_DATA_PATH = os.path.join(_SKIPPD_DATA_DIR, "skippd_train.parquet")
OUTPUT_DATA_PATH = os.path.join(_SKIPPD_DATA_DIR, "skippd_train_cleaned_30min_with_images_v12_correct_timezone.parquet")

# --- Time & Frequency Settings ---
TIMEZONE = "America/Los_Angeles"
FREQ = "30min"

# Temporal bounds for the dataset. Data outside this range is discarded.
START_DATE = "2017-05-08 07:30:00-07:00"
CUTOFF_DATE = "2018-11-03 18:37:10-07:00"

# --- Imputation Strategy Thresholds ---
LARGE_GAP_DAYS = 7
medium_gap_days = 8 
SMALL_GAP_HOURS = 25
ULTRA_SHORT_GAP_HOURS = 4

# --- Night Time Logic ---
NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 8

# --- Ad-Hoc Data Corrections ---
# Specific interventions for known data quality issues identified during EDA.
# Each fix specifies a time window and the repair type ('zero' or 'ffill').
AD_HOC_FIXES = [
    {
        "name": "Fix 1 (May 10 Missing Data)",
        "start": "2017-05-10 11:00:00-07:00",
        "end": "2017-05-10 20:00:00-07:00",
        "type": "zero"
    },
    {
        "name": "Fix 2 (March 9 Missing Data)",
        "start": "2017-03-09 18:00:00-08:00",
        "end": "2017-03-09 20:00:00-08:00",
        "type": "zero"
    },
    {
        "name": "Fix 3 (March 10 Morning Gap)",
        "start": "2017-03-10 07:30:00-08:00",
        "end": "2017-03-10 09:30:00-08:00",
        "type": "ffill"
    }
]
