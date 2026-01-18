import pandas as pd
import torch

def load_data(path):
    print(f"Loading dataset from {path}...")
    df = pd.read_parquet(path)
    
    if 'time' in df.columns and 'timestamp' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['time']) 
        if df['timestamp'].dt.tz is not None:
             df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    if "series_id" in df.columns:
        df = df.rename(columns={"series_id": "item_id", "pv": "pv_value"})

    # Drop 'time' if it exists to avoid type errors in Chronos pipeline (which dislikes TZ-aware columns)
    if 'time' in df.columns:
        df = df.drop(columns=['time'])

    df = df.sort_values(['item_id', 'timestamp']).reset_index(drop=True)
    return df

def identify_covariates(df):
    reserved_columns = ['timestamp', 'item_id', 'pv_value', 'visual_embedding']
    cov_cols = [col for col in df.columns if col not in reserved_columns]
    print(f"Identified {len(cov_cols)} covariates: {cov_cols}")
    return cov_cols

def split_ts_dataset(df, prediction_length):
    # Filter out series that are too short
    item_counts = df.groupby('item_id').size()
    valid_items = item_counts[item_counts > prediction_length].index

    if len(valid_items) < len(item_counts):
        print(f"Dropping {len(item_counts) - len(valid_items)} series that are too short.")
        df = df[df['item_id'].isin(valid_items)].copy()

    # Sort
    df = df.sort_values(['item_id', 'timestamp']).reset_index(drop=True)

    print(f"Splitting data for {len(valid_items)} time series...")

    # Inference/Test: Grab the last prediction_length rows for EACH item_id
    test_df = df.groupby('item_id').tail(prediction_length).copy()

    # Train: Drop the rows that belong to test_df
    train_df = df.drop(test_df.index).copy()

    # Inference input (Drop target)
    inference_df = test_df.copy()
    if 'pv_value' in inference_df.columns:
        inference_df = inference_df.drop(columns=['pv_value'])

    print(f"Train shape: {train_df.shape}")
    print(f"Inference/Test shape: {inference_df.shape}")

    return train_df, inference_df, test_df
