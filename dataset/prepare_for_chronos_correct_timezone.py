import pandas as pd
import numpy as np
import io
from PIL import Image
import config 

class DatasetPreparer:
    """
    Handles the end-to-end preparation of the dataset for Chronos training.
    
    Responsibilities:
    1. Loading and cleaning raw parquet data.
    2. Resampling time series to a strict grid (e.g., 30min).
    3. Imputing missing values using a tiered strategy (Forward, Day-Prior, Week-Prior).
    4. Applying domain-specific logic (e.g., Night Time Zero-Fill).
    5. Segmenting the continuous timeline into valid training sequences.
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.black_img_dict = self._create_black_image_dict()

    def _create_black_image_dict(self):
        """Creates a dummy black image artifact for missing visual data."""
        img = Image.new('RGB', (64, 64), color='black')
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        return {'bytes': buf.getvalue(), 'path': None}

    def load_data(self, path):
        print(f"Loading dataset from {path}...")
        try:
            return pd.read_parquet(path)
        except Exception as e:
            print(f"Critical Error: Failed to load dataset. {e}")
            return None

    def clean_and_resample(self, df):
        """
        Standardizes the time index:
        - Detects timestamp column.
        - Truncates to the configured date range.
        - Resamples to a strict frequency grid, introducing NaNs for missing steps.
        """
        # Column Detection
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if not timestamp_cols:
            raise ValueError("Schema Error: No valid timestamp column found in dataset.")
        
        time_col = timestamp_cols[0]
        print(f"Detected timestamp column: '{time_col}'")

        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])

        # Temporal Filtration
        start_date = pd.Timestamp(self.cfg.START_DATE)
        cutoff_date = pd.Timestamp(self.cfg.CUTOFF_DATE)
        
        print(f"Filtering data to range: {start_date} -> {cutoff_date}...")
        df = df[(df[time_col] >= start_date) & (df[time_col] <= cutoff_date)].copy()

        # Grid Alignment
        # Round timestamps to the nearest frequency step to correct minor drift
        df['time_rounded'] = df[time_col].dt.round(self.cfg.FREQ)
        # Drop duplicates resulting from rounding (keep first occurrence)
        df = df.drop_duplicates(subset=['time_rounded'], keep='first')
        df = df.set_index('time_rounded').sort_index()

        # Strict Resampling
        # This creates explicit rows for every timestamp in the grid, filling missing ones with NaNs
        print(f"Resampling to strict {self.cfg.FREQ} frequency grid...")
        df_resampled = df.resample(self.cfg.FREQ).asfreq()

        # Boundary Restoration
        # Ensure the dataset starts exactly at START_DATE, even if the first data point is later
        if df_resampled.index.min() > start_date:
            target_start = start_date
            if df_resampled.index.tz is not None:
                target_start = start_date.tz_convert(df_resampled.index.tz)
            
            print(f"Restoring dataset start boundary: {target_start}")
            new_index = pd.date_range(start=target_start, end=df_resampled.index.max(), freq=self.cfg.FREQ)
            df_resampled = df_resampled.reindex(new_index)

        return df_resampled

    def impute_data(self, df):
        """
        Executes the imputation strategy.
        
        Strategy Overview:
        1. Identify all gaps (consecutive NaNs).
        2. "Night Time Fill": Fill gaps occurring at night (20:00-08:00) with 0.
        3. Tiered Imputation on remaining gaps:
           - Ultra Short (<4h): Forward Fill
           - Short (<25h): Copy from previous day
           - Medium (<1 week): Copy from previous week
           - Large: Leave as NaN (will split the series later) or attempt Yearly fallback.
        """
        print("Initializing Imputation Pipeline...")
        check_col = df.columns[0]
        
        # Initial Gap Analysis
        is_missing = df[check_col].isna()
        gap_groups = is_missing.ne(is_missing.shift()).cumsum()
        gap_groups = gap_groups[is_missing]
        
        # Pre-calculate Mask for "Large Gaps" to prevent improper filling
        # We generally do NOT want to synthesize data for week-long outages.
        large_gap_mask = pd.Series(False, index=df.index)
        for gap_id in gap_groups.unique():
            indices = gap_groups[gap_groups == gap_id].index
            duration = (indices[-1] - indices[0]) + pd.Timedelta(self.cfg.FREQ)
            if duration > pd.Timedelta(days=self.cfg.LARGE_GAP_DAYS):
                large_gap_mask.loc[indices] = True

        # --- Tier 0: Night Time Zero-Fill ---
        # Domain Logic: PV generation is 0 at night. We can safely fill missing night data.
        self._night_time_fill(df, is_missing, large_gap_mask)

        # Re-evaluate gaps now that nights are potentially filled
        is_missing = df[check_col].isna()
        gap_groups = is_missing.ne(is_missing.shift()).cumsum()
        gap_groups = gap_groups[is_missing]
        
        unique_gaps = gap_groups.unique()
        print(f"Processing {len(unique_gaps)} remaining data gaps...")

        for gap_id in unique_gaps:
            gap_indices = gap_groups[gap_groups == gap_id].index
            duration = (gap_indices[-1] - gap_indices[0]) + pd.Timedelta(self.cfg.FREQ)
            
            # Select Imputation Strategy based on Gap Duration
            if duration <= pd.Timedelta(hours=self.cfg.ULTRA_SHORT_GAP_HOURS):
                 self._fill_forward(df, gap_indices)
            elif duration < pd.Timedelta(hours=self.cfg.SMALL_GAP_HOURS):
                 self._fill_day_prior(df, gap_indices)
            elif duration < pd.Timedelta(days=self.cfg.medium_gap_days):
                 self._fill_week_prior(df, gap_indices)
            else:
                 # Large gaps usually remain NaN, but we try a yearly fallback just in case
                 self._fill_yearly(df, gap_indices)
        
        return df

    def _night_time_fill(self, df, is_missing, large_gap_mask):
        """Fills missing values with 0 if they fall within the designated night hours."""
        print("Running Night Time Zero-Fill...")
        
        # Ensure we check hours in the correct Local Timezone
        if df.index.tz is None:
            check_index = df.index.tz_localize(self.cfg.TIMEZONE, ambiguous='NaT', nonexistent='NaT')
        else:
            check_index = df.index.tz_convert(self.cfg.TIMEZONE)
            
        night_mask = (check_index.hour >= self.cfg.NIGHT_START_HOUR) | (check_index.hour < self.cfg.NIGHT_END_HOUR)
        
        # Only fill if: Missing AND Night AND NOT a huge gap (safety)
        to_fill = is_missing & night_mask & (~large_gap_mask)
        
        if to_fill.any():
            self._apply_zero_fill(df, to_fill)
            print(f"  -> Filled {to_fill.sum()} timestamps.")

    def _apply_zero_fill(self, df, mask):
        """Helper to fill numerical cols with 0 and image cols with the black image."""
        cols = [c for c in df.columns if c != 'series_id']
        count = mask.sum()
        for c in cols:
            if c == 'image':
                 df.loc[mask, c] = pd.Series([self.black_img_dict] * count, index=df[mask].index)
            else:
                 df.loc[mask, c] = 0.0

    def _fill_forward(self, df, indices):
        """Tier 1: Simply copy the last valid value (Good for very short dropouts)."""
        offset = pd.Timedelta(self.cfg.FREQ)
        for t in indices:
            prev_t = t - offset
            if prev_t in df.index:
                df.loc[t] = df.loc[prev_t]

    def _fill_day_prior(self, df, indices):
        """Tier 2: Copy values from exactly 24h ago (captures daily seasonality)."""
        offset = pd.Timedelta(hours=24)
        for t in indices:
            found = False
            # Look back up to 7 days to find a valid source
            for day_back in range(1, 8):
                source_t = t - (offset * day_back)
                if source_t in df.index and not df.loc[source_t].isna().all():
                     df.loc[t] = df.loc[source_t]
                     found = True
                     break
            if not found: self._fill_yearly(df, [t]) # Fallback

    def _fill_week_prior(self, df, indices):
        """Tier 3: Copy values from 1 week ago (captures weekly seasonality)."""
        offset = pd.Timedelta(days=7)
        for t in indices:
            found = False
            prev = t - offset
            if prev in df.index and not df.loc[prev].isna().all():
                df.loc[t] = df.loc[prev]
                found = True
            
            # Try next week (future) if past is missing
            if not found:
                nxt = t + offset
                if nxt in df.index and not df.loc[nxt].isna().all():
                    df.loc[t] = df.loc[nxt]

    def _fill_yearly(self, df, indices):
        """Tier 4 / Fallback: Copy values from 1 year ago."""
        for t in indices:
            found = False
            try:
                prev_yr = t - pd.DateOffset(years=1)
                if prev_yr in df.index and not df.loc[prev_yr].isna().all():
                    df.loc[t] = df.loc[prev_yr]
                    found = True
            except: pass
            
            if not found:
                 try:
                    next_yr = t + pd.DateOffset(years=1)
                    if next_yr in df.index and not df.loc[next_yr].isna().all():
                        df.loc[t] = df.loc[next_yr]
                 except: pass

    def apply_adhoc_fixes(self, df):
        """Applies manual patches defined in configuration for specific data corruptions."""
        print("Applying Ad-Hoc Fixes from Config...")
        for fix in self.cfg.AD_HOC_FIXES:
            try:
                start = pd.Timestamp(fix['start'])
                end = pd.Timestamp(fix['end'])
                mask = (df.index >= start) & (df.index <= end)
                
                if mask.any():
                    if fix['type'] == 'zero':
                        self._apply_zero_fill(df, mask)
                    elif fix['type'] == 'ffill':
                        df.loc[mask] = df.loc[mask].ffill()
                    
                    print(f"  -> Applied {fix['name']} ({fix['type']})")
            except Exception as e:
                print(f"  -> Warning: Failed to apply {fix['name']}: {e}")
        return df

    def segment_and_save(self, df, output_path):
        """
        Finalizes the dataset:
        1. Breaks continuity where NaNs still exist (creates new series IDs).
        2. Renames columns and resets index.
        3. Saves to Parquet.
        """
        print("Segmenting continuous series...")
        # A new series ID is generated whenever a NaN block is encountered
        valid_mask = ~df.isna().any(axis=1)
        df['series_id'] = (valid_mask != valid_mask.shift()).cumsum()
        
        # Keep only valid rows
        df_final = df[valid_mask].copy()
        
        # Renumber Series IDs nicely (0, 1, 2...)
        unique_ids = df_final['series_id'].unique()
        id_map = {uid: i for i, uid in enumerate(unique_ids)}
        df_final['series_id'] = df_final['series_id'].map(id_map)
        
        print(f"Generated {len(unique_ids)} independent time series segments.")
        
        df_final = df_final.reset_index().rename(columns={'time_rounded': 'time'})
        
        print(f"Saving processed dataset to: {output_path}")
        df_final.to_parquet(output_path)
        print("Success.")

def main():
    preparer = DatasetPreparer(config)
    
    df = preparer.load_data(config.RAW_DATA_PATH)
    if df is None: return

    # Pipeline Execution
    df = preparer.clean_and_resample(df)
    df = preparer.impute_data(df)
    df = preparer.apply_adhoc_fixes(df)
    preparer.segment_and_save(df, config.OUTPUT_DATA_PATH)

if __name__ == "__main__":
    main()
