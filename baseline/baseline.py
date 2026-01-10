import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import os
from utils import plot_prediction

# --- Configuration ---
COVARIATE_COLUMNS = ['time_hour_sin', 'time_hour_cos', 'time_dayofyear_sin']
BASE_DIR = "/content/drive/MyDrive/FM_project/dataset"
TRAIN_PATH = os.path.join(BASE_DIR, "skippd_train_aligned_v13_with_time_features_no_images.parquet")
PREDICTION_LENGTH = 96
RESULTS_DIR = "results"
USE_CV2_SAVED_MODEL = False

def load_and_prepare(path):
    print(f"Loading {path}...")
    df = pd.read_parquet(path)

    # Rename columns to match AutoGluon's expected format
    column_mapping = {
        "time": "timestamp",
        "series_id": "item_id",
        "pv": "pv_value"
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Timestamp Conversion & Remove Timezone
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is not None:
        print("Detected timezone info. Removing for AutoGluon compatibility...")
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    missing_covariates = [col for col in COVARIATE_COLUMNS if col not in df.columns]
    if missing_covariates:
        raise ValueError(f"Missing required covariate columns: {missing_covariates}")

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
      enable_ensemble=False, # Disable ensemble to isolate Chronos performance
      random_seed=42
  )

  if use_saved_models:
    return train_data, bolt_predictor, None

  # Initialize the Predictor for Chronos-2
  c2_predictor = TimeSeriesPredictor(
      prediction_length=PREDICTION_LENGTH,
      target="pv_value",
      path="autogluon_chronos2_results", # Folder to save models
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
      else:
          print(f"Warning: Model '{model_name}' not found in predictor.")

  return predictions_dict

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    full_df = load_and_prepare(TRAIN_PATH)
    print(f"Data shape: {full_df.shape}")

    # Fit models (Returns full trained data for context)
    train_data, bolt_predictor, c2_predictor = fit_model(full_df, use_saved_models=USE_CV2_SAVED_MODEL)

    # Generate Predictions 
    past_data = train_data.slice_by_timestep(None, -PREDICTION_LENGTH)
    known_covariates_future = train_data[COVARIATE_COLUMNS]

    # Predict with Chronos-2
    print("Generating Chronos-2 predictions...")
    cv2_model_predictions = chronos2prediction(
        past_data, 
        known_covariates_future, 
        known_covariates=known_covariates_future, 
        c2_predictor=c2_predictor, 
        use_saved_models=USE_CV2_SAVED_MODEL
    )

    # Plot and Save Results
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
