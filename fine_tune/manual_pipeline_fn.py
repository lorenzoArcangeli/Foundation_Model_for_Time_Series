import pandas as pd
import torch
import numpy as np
import os
from chronos import BaseChronosPipeline, Chronos2Pipeline
from peft import LoraConfig
from utils.plotting_utils import calculate_item_mase, calculate_item_mape, calculate_item_wmape, plot_model_comparison
from utils.cv import backtest_model

# --- Configuration ---
ADAPTER_TYPE = "dora" # Options: "lora", "dora"
PREDICTION_LENGTH = 96
BATCH_SIZE = 12 
LEARNING_RATE = 1e-4
NUM_STEPS = 200
LOGGING_STEPS = 100
BASE_DIR = "/content/drive/MyDrive/FM_project/dataset"
RESULTS_DIR = "results"

TRAIN_PATH = os.path.join(BASE_DIR, "skippd_train_aligned_v13_with_time_features_and_sky_features.parquet")

def load_and_prepare(path):
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    if "image" in df.columns:
      print("Dropping raw 'image' column (dictionaries)...")
    df = df.drop(columns=["image"])

    # Rename columns
    column_mapping = {
        "time": "timestamp",
        "series_id": "item_id",
        "pv": "pv_value"
    }

    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Timestamp Conversion
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    reserved_columns = ['timestamp', 'item_id', 'pv_value']

    covariate_columns = [col for col in df.columns if col not in reserved_columns]

    print(f" Automatically identified {len(covariate_columns)} covariates: {covariate_columns}")

    # Check Covariates
    missing_covariates = [col for col in covariate_columns if col not in df.columns]
    if missing_covariates:
        raise ValueError(f"Missing required covariate columns: {missing_covariates}")

    # Filter out series that are too short
    item_counts = df.groupby('item_id').size()
    valid_items = item_counts[item_counts > PREDICTION_LENGTH].index

    if len(valid_items) < len(item_counts):
        print(f"Dropping {len(item_counts) - len(valid_items)} series that are too short.")
        df = df[df['item_id'].isin(valid_items)].copy()

    # Sort by item_id AND timestamp
    df = df.sort_values(['item_id', 'timestamp']).reset_index(drop=True)

    # Global Split
    print(f"Splitting data for {len(valid_items)} time series...")

    test_df = df.groupby('item_id').tail(PREDICTION_LENGTH).copy()

    train_df = df.drop(test_df.index).copy()

    inference_df = test_df.copy()
    if 'pv_value' in inference_df.columns:
        inference_df = inference_df.drop(columns=['pv_value'])

    print(f"Train shape: {train_df.shape}")
    print(f"Inference/Test shape: {inference_df.shape}")

    return train_df, inference_df, test_df, covariate_columns

def prepare_train_inputs(train_df, covariate_columns, target_col="pv_value"):
    train_inputs = []

    known_covariates = covariate_columns
    
    for item_id, group in train_df.groupby("item_id"):
        covariates_dict = {col: group[col].values for col in covariate_columns}
        
        train_inputs.append({
            "target": group[target_col].values,
            "past_covariates": covariates_dict,
            "future_covariates": {col: None for col in known_covariates} 
        })
    return train_inputs


def main():
    # Initialize Pipeline
    print("Initializing Chronos Pipeline...")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2", 
        device_map="cuda", 
        torch_dtype=torch.bfloat16
    )

    # Load Data
    train_df, inference_df, test_df, covariate_columns = load_and_prepare(TRAIN_PATH)

    # Zero-Shot Prediction (Baseline)
    print("Running Zero-Shot (Base) predictions...")
    pred_df = pipeline.predict_df(
        df=train_df,
        future_df=inference_df,
        prediction_length=PREDICTION_LENGTH,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="item_id",
        timestamp_column="timestamp",
        target="pv_value",
    )

    # Fine-Tuning Setup
    print("Preparing training inputs...")
    train_inputs = prepare_train_inputs(train_df, covariate_columns)

    print(f"Configuring Adapter: {ADAPTER_TYPE.upper()}")
    
    peft_config = None
    if ADAPTER_TYPE == "dora":
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "self_attention.q", "self_attention.v", 
                "self_attention.k", "self_attention.o", 
                "output_patch_embedding.output_layer"
            ],
            lora_dropout=0.05,
            bias="none",
            use_dora=True 
        )
    elif ADAPTER_TYPE == "lora":
         peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "self_attention.q", "self_attention.v", 
                "self_attention.k", "self_attention.o", 
                "output_patch_embedding.output_layer"
            ],
            lora_dropout=0.05,
            bias="none",
        )
    
    # Fine-tune
    print("Starting Fine-Tuning...")
    lora_pipeline = pipeline.fit(
        inputs=train_inputs,
        prediction_length=PREDICTION_LENGTH,
        num_steps=NUM_STEPS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        logging_steps=LOGGING_STEPS,
        finetune_mode=ADAPTER_TYPE,
        lora_config=peft_config if ADAPTER_TYPE == "dora" else None 
    )

    print("Running Fine-Tuned predictions...")
    lora_pred_df = lora_pipeline.predict_df(
        df=train_df,
        future_df=inference_df,
        prediction_length=PREDICTION_LENGTH,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="item_id",
        timestamp_column="timestamp",
        target="pv_value",
    )

    # Backtesting
    full_df = pd.concat([train_df, test_df]).sort_values(['item_id', 'timestamp'])
    backtest_model(
        pipeline=lora_pipeline,
        df=full_df,
        num_windows=10,
        step_size=PREDICTION_LENGTH
    )

    # Plotting
    models_to_plot = {
        "Zero-Shot (Base)": pred_df,
        f"Fine-Tuned ({ADAPTER_TYPE.upper()})": lora_pred_df,
    }
    
    print(f"Saving plots to {RESULTS_DIR}...")
    plot_model_comparison(
        train_df=train_df,
        test_df=test_df,
        model_predictions=models_to_plot,
        plot_history_length=200, 
        prediction_length=PREDICTION_LENGTH,
        seasonality=96,
        save_dir=RESULTS_DIR
    )

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()