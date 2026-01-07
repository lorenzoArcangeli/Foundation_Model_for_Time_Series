import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from chronos import BaseChronosPipeline, Chronos2Pipeline
from peft import LoraConfig
import itertools


pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda", torch_dtype=torch.bfloat16)

PREDICTION_LENGTH = 96
base_dir = DATA_PATH = "/content/drive/MyDrive/FM_project/dataset"
train_path = os.path.join(base_dir, "skippd_train_aligned_v13_with_time_features_and_sky_features.parquet")

COVARIATE_COLUMNS = list()

def load_and_prepare(path):
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    if "image" in df.columns:
      print("Dropping raw 'image' column (dictionaries)...")
    df = df.drop(columns=["image"])

    # 1. Rename columns
    column_mapping = {
        "time": "timestamp",
        "series_id": "item_id",
        "pv": "pv_value"
    }


    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # 2. Timestamp Conversion
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    reserved_columns = ['timestamp', 'item_id', 'pv_value']

    COVARIATE_COLUMNS = [col for col in df.columns if col not in reserved_columns]

    print(f" Automatically identified {len(COVARIATE_COLUMNS)} covariates: {COVARIATE_COLUMNS}")

    # 3. Check Covariates
    missing_covariates = [col for col in COVARIATE_COLUMNS if col not in df.columns]
    if missing_covariates:
        raise ValueError(f"Missing required covariate columns: {missing_covariates}")

    # --- UPDATED LOGIC FOR MULTIPLE SERIES ---

    # 4. Filter out series that are too short
    # We need at least prediction_length + 1 data point to have a training set
    item_counts = df.groupby('item_id').size()
    valid_items = item_counts[item_counts > PREDICTION_LENGTH].index

    if len(valid_items) < len(item_counts):
        print(f"Dropping {len(item_counts) - len(valid_items)} series that are too short.")
        df = df[df['item_id'].isin(valid_items)].copy()

    # 5. Sort by item_id AND timestamp (Critical for correct splitting)
    df = df.sort_values(['item_id', 'timestamp']).reset_index(drop=True)

    # 6. Global Split
    print(f"Splitting data for {len(valid_items)} time series...")

    # Inference/Test: Grab the last PREDICTION_LENGTH rows for EACH item_id
    test_df = df.groupby('item_id').tail(PREDICTION_LENGTH).copy()

    # Train: Drop the rows that belong to test_df
    # (Since we reset_index above, the indices are unique and safe to use for dropping)
    train_df = df.drop(test_df.index).copy()

    # 7. Create Inference input (Drop target)
    inference_df = test_df.copy()
    if 'pv_value' in inference_df.columns:
        inference_df = inference_df.drop(columns=['pv_value'])

    print(f"Train shape: {train_df.shape}")
    print(f"Inference/Test shape: {inference_df.shape}")

    return train_df, inference_df, test_df

target = "pv_value"  
prediction_length = PREDICTION_LENGTH  
id_column = "item_id"  
timestamp_column = "timestamp"
timeseries_id = 0

train_df, inference_df, test_df =load_and_prepare(train_path)

pred_df = pipeline.predict_df(
    df=train_df,
    future_df=inference_df,
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column=id_column,
    timestamp_column=timestamp_column,
    target=target,
)



# Prepare data for fine-tuning using the retail sales dataset
known_covariates = COVARIATE_COLUMNS

train_inputs = []
for item_id, group in train_df.groupby("item_id"):

    covariates_dict = {col: group[col].values for col in COVARIATE_COLUMNS}

    train_inputs.append({
        "target": group[target].values,
        "past_covariates": covariates_dict,#{col: group[col].values for col in past_covariates + known_covariates},
        # Future values of covariates are not used during training.
        # However, we need to include their names to indicate that these columns will be available at prediction time
        #"future_covariates": covariates_dict,#{col: None for col in known_covariates},
        "future_covariates": {col: None for col in known_covariates}
    })


# Fine-tune the model with LoRA or DoRA
lora_finetuned_pipeline = pipeline.fit(
    inputs=train_inputs,
    prediction_length=PREDICTION_LENGTH,
    num_steps=200,
    learning_rate=1e-4,
    batch_size=24, # I use 12 with Dora
    logging_steps=100,
    finetune_mode="lora",
    #lora_config=dora_config,
)


lora_finetuned_pred_df = lora_finetuned_pipeline.predict_df(
    df=train_df,
    future_df=inference_df,
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column=id_column,
    timestamp_column=timestamp_column,
    target=target,
)
