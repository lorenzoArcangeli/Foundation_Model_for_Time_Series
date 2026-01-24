import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import os
from utils import plot_prediction

# --- Configuration ---
USE_IMAGE_FEATURES = False # Toggle this to True to use image features
PREDICTION_LENGTH = 96
RESULTS_DIR = "results"
USE_CV2_SAVED_MODEL = False

BASE_DIR = "/content/drive/MyDrive/FM_project/dataset"

# Dataset Paths
TRAIN_PATH_NO_IMAGES = os.path.join(BASE_DIR, "skippd_train_aligned_v13_with_time_features_no_images.parquet")
TRAIN_PATH_WITH_IMAGES = os.path.join(BASE_DIR, "skippd_train_aligned_v13_with_time_features_and_sky_features.parquet")

# Covariates
COVARIATE_COLUMNS = ['time_hour_sin', 'time_hour_cos', 'time_dayofyear_sin']

if USE_IMAGE_FEATURES:
    TRAIN_PATH = TRAIN_PATH_WITH_IMAGES
    SKY_FEATURE_COLS = [f"sky_feature_{i}" for i in range(10)]
    COVARIATE_COLUMNS.extend(SKY_FEATURE_COLS)
else:
    TRAIN_PATH = TRAIN_PATH_NO_IMAGES
    SKY_FEATURE_COLS = []

def load_and_prepare(path):
    print(f"Loading {path}...")
    df = pd.read_parquet(path)

    # Rename columns to match AutoGluon format
    column_mapping = {
        "time": "timestamp",
        "series_id": "item_id",
        "pv": "pv_value"
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is not None:
        # remove for autogluon compatibility
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    missing_covariates = [col for col in COVARIATE_COLUMNS if col not in df.columns]
    if missing_covariates:
        print(f"Columns in DF: {df.columns}")
        raise ValueError(f"Missing required covariate columns: {missing_covariates}")

    # Handle Sky Features (if applicable)
    if USE_IMAGE_FEATURES and SKY_FEATURE_COLS:
        # Just to check
        if df[SKY_FEATURE_COLS].isnull().values.any():
            print("Warning: NaNs found in sky features. Filling with 0.")
            df[SKY_FEATURE_COLS] = df[SKY_FEATURE_COLS].fillna(0)
            
    # Create TimeSeriesDataFrame
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="timestamp"
    )

    for col in COVARIATE_COLUMNS:
        ts_df[col] = ts_df[col].astype(float)

    print(f"TimeSeriesDataFrame created. Shape: {ts_df.shape}")
    return ts_df

def fit_model(full_df, use_saved_models=USE_CV2_SAVED_MODEL):
  train_data = TimeSeriesDataFrame.from_data_frame(
    full_df,
    id_column="item_id",
    timestamp_column="timestamp"
  )

  # Verify frequency
  print("\nVerifying Data Integrity...")
  train_data = train_data.convert_frequency(freq='30min')

  # Try different Chronos models
  model_hyperparameters = {
      "Chronos": [
          {
          "model_path": "amazon/chronos-bolt-small",
          "batch_size": 32,
          "context_length": 512,
          "optimization.max_epochs": 10, 
      },
      {
          "model_path": "amazon/chronos-bolt-base",
          "batch_size": 16,
          "context_length": 512,
          "ag_args": {"name_suffix": "ZeroShot"},
          "optimization.max_epochs": 10, 
      },
      {
          "model_path": "amazon/chronos-bolt-base",
          "batch_size": 16,
          "context_length": 512,
          "covariate_regressor": "CAT",
          "target_scaler": "standard",
          "ag_args": {"name_suffix": "WithRegressor"},
          "optimization.max_epochs": 10,
      }
      ]
  }

  bolt_predictor = TimeSeriesPredictor(
      prediction_length=PREDICTION_LENGTH,
      path="autogluon_chronos_pv_forecast",
      target="pv_value",
      eval_metric="MASE",
      known_covariates_names=COVARIATE_COLUMNS
  )

  print("\nStarting Training with Bolt model(s)")
  bolt_predictor.fit(
      train_data,
      hyperparameters=model_hyperparameters,
      enable_ensemble=False,
      random_seed=42
  )

  if use_saved_models:
    return train_data, bolt_predictor, None

  # Initialize the Predictor for Chronos-2
  c2_predictor = TimeSeriesPredictor(
      prediction_length=PREDICTION_LENGTH,
      target="pv_value",
      path="autogluon_chronos2_results",
      eval_metric="MASE", 
      known_covariates_names=COVARIATE_COLUMNS
  )

  robust_hyperparameters = {
    "Chronos2": [  
        {
            "ag_args": {"name_suffix": "ZeroShot"}
        }
    ]
  }

  print("Running Chronos-2 with visible fine-tuning...")
  c2_predictor.fit(
      full_df,
      hyperparameters=robust_hyperparameters,
      num_val_windows=2,
      val_step_size=PREDICTION_LENGTH,
      verbosity=2  
  )

  return train_data, bolt_predictor, c2_predictor


def chronos2prediction(past_data, known_covariates_future, known_covariates, c2_predictor, use_saved_models=True):
  if not use_saved_models:
    cv2_model_names = c2_predictor.model_names()
    cv2_model_predictions = {}

    for model_name in cv2_model_names:
        print(f"Predicting with {model_name}...")
        cv2_model_predictions[model_name] = c2_predictor.predict(past_data, known_covariates=known_covariates_future, model=model_name)
    return cv2_model_predictions


  predictor = TimeSeriesPredictor.load("restored_right/autogluon_chronos2_results")
  target_models = [
      "Chronos2ZeroShot",
  ]
  predictions_dict = {}

  print("Available models in this predictor:", predictor.model_names())

  for model_name in target_models:
      if model_name in predictor.model_names():
          print(f"Generating forecast for: {model_name}...")

          preds = predictor.predict(
              past_data,
              known_covariates=known_covariates_future,
              model=model_name
          )

          predictions_dict[model_name] = preds

          # Optional: Save each to CSV
          preds.to_csv(os.path.join(RESULTS_DIR, f"forecast_{model_name}.csv"))
      else:
          print(f"Warning: Model '{model_name}' not found in predictor.")

  return predictions_dict

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    full_df = load_and_prepare(TRAIN_PATH)
    
    # Drop the raw image column if present
    if "image" in full_df.columns:
        print("Dropping raw 'image' column (dictionaries)...")
        full_df = full_df.drop(columns=["image"])

    print(f"Data shape: {full_df.shape}")

    # Fit models
    train_data, bolt_predictor, c2_predictor = fit_model(full_df, use_saved_models=USE_CV2_SAVED_MODEL)

    # Generate Predictions
    past_data = train_data.slice_by_timestep(None, -PREDICTION_LENGTH)
    known_covariates_future = train_data[COVARIATE_COLUMNS]

    print("Generating Chronos-2 predictions...")
    cv2_model_predictions = chronos2prediction(
        past_data, 
        known_covariates_future, 
        known_covariates=known_covariates_future, 
        c2_predictor=c2_predictor, 
        use_saved_models=USE_CV2_SAVED_MODEL
    )

    print(f"Saving plots to {RESULTS_DIR}...")
    plot_prediction(
        past_data, 
        train_data, 
        bolt_predictor, 
        cv2_model_predictions, 
        known_covariates_future,
        save_dir=RESULTS_DIR
    )

if __name__ == "__main__":
    main()
