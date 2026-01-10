import pandas as pd
import torch
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt
import os

COVARIATE_COLUMNS = ['time_hour_sin', 'time_hour_cos', 'time_dayofyear_sin']
base_dir = DATA_PATH = "/content/drive/MyDrive/FM_project/dataset"
#train_path = os.path.join(base_dir, "skippd_train_cleaned_30min_no_images_v12.parquet")
train_path = os.path.join(base_dir, "skippd_train_aligned_v13_with_time_features_no_images.parquet")
PREDICTION_LENGTH = 96
USE_CV2_SAVED_MODEL=False

def load_and_prepare(path):
    print(f"Loading {path}...")
    df = pd.read_parquet(path)

    # 1. Inspect initial columns (for debugging)
    print(f"Original columns: {list(df.columns)}")

    # 2. Rename columns to match AutoGluon's expected format
    # Map 'time' -> 'timestamp' and 'series_id' -> 'item_id'
    column_mapping = {
        "time": "timestamp",
        "series_id": "item_id",
        # If your target is 'pv', we can keep it or rename to 'target'/'pv_value'
        "pv": "pv_value"
    }

    # Only rename columns that actually exist
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # 3. Force Timestamp Conversion & Remove Timezone
    # AutoGluon requires naive datetime64[ns]
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if df['timestamp'].dt.tz is not None:
        print("Detected timezone info. Removing for AutoGluon compatibility...")
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    print(f"Final columns: {list(df.columns)}")
    print(f"Timestamp type: {df['timestamp'].dtype}")

    missing_covariates = [col for col in COVARIATE_COLUMNS if col not in df.columns]
    if missing_covariates:
        raise ValueError(f"Missing required covariate columns: {missing_covariates}")

    # 4. Create TimeSeriesDataFrame
    # Now we are sure 'item_id' and 'timestamp' exist
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="timestamp"
    )

    for col in COVARIATE_COLUMNS:
        ts_df[col] = ts_df[col].astype(float)

    print(f"TimeSeriesDataFrame created. Shape: {ts_df.shape}")
    print(f"Included features: {list(ts_df.columns)}")

    return ts_df

def fit_model(full_df, use_saved_models=USE_CV2_SAVED_MODEL):
  train_data = TimeSeriesDataFrame.from_data_frame(
    full_df,
    id_column="item_id",
    timestamp_column="timestamp"
  )

  # Verify frequency and check for internal gaps (should be none within segments)
  print("\nVerifying Data Integrity...")
  train_data = train_data.convert_frequency(freq='30min')
  print(train_data.head())

  # Define prediction horizon (e.g., 2 days = 48 hours = 96 steps)
  PREDICTION_LENGTH = 96
  model_hyperparameters = {
      "Chronos": [
          {
          "model_path": "amazon/chronos-bolt-small",
          "batch_size": 32,
          "context_length": 512,
          "optimization.max_epochs": 10, # Increase for better results
      },
      {
          "model_path": "amazon/chronos-bolt-base",
          "batch_size": 16,
          "context_length": 512,
          "ag_args": {"name_suffix": "ZeroShot"},
          "optimization.max_epochs": 10, # Increase for better results
      },
      {
          "model_path": "amazon/chronos-bolt-base",
          "batch_size": 16,
          "context_length": 512,
          "covariate_regressor": "CAT",
          "target_scaler": "standard",
          "ag_args": {"name_suffix": "WithRegressor"},
          "optimization.max_epochs": 10, # Increase for better results
      }
      ]
  }

  bolt_predictor = TimeSeriesPredictor(
      prediction_length=PREDICTION_LENGTH,
      path="autogluon_chronos_pv_forecast",
      target="pv_value",
      eval_metric="MASE", # Mean Absolute Scaled Error
      #freq='30min',
      known_covariates_names=COVARIATE_COLUMNS
  )

  print("\nStarting Training with Bolt model(s)")
  bolt_predictor.fit(
      train_data,
      hyperparameters=model_hyperparameters,
      enable_ensemble=False, # Disable ensemble to isolate Chronos performance
      random_seed=42
  )

  # IF I USED SAVED MODELS FOR CHRONOS 2
  if use_saved_models:
    return train_data, bolt_predictor, None

  # 3. Initialize the Predictor
  # We must specify 'target="pv_value"' because you renamed the 'pv' column.
  c2_predictor = TimeSeriesPredictor(
      prediction_length=PREDICTION_LENGTH,
      target="pv_value",
      path="autogluon_chronos2_results", # Folder to save models
      eval_metric="MASE", # Mean Absolute Scaled Error (common for forecasting)
      known_covariates_names=COVARIATE_COLUMNS
  )

  robust_hyperparameters = {
    "Chronos2": [  # Use "Chronos" as the key for all Chronos variants (including Chronos-2)
        # 1. Zero-shot version (Baseline)
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
      verbosity=2  # Increase verbosity to 3 or 4 to ensure underlying logs pass through
  )

  return train_data, bolt_predictor, c2_predictor

def calculate_item_mase(y_true, y_pred, y_history):
    """
    Calculates MASE for a single item.
    MASE = MAE(forecast) / MAE(naive_history)
    """
    # 1. Calculate MAE of the forecast (Numerator)
    # Ensure alignment
    mae_forecast = np.mean(np.abs(y_true.values - y_pred.values))

    # 2. Calculate MAE of naive forecast on history (Denominator)
    # We use lag-1 (standard naive) for the denominator
    if len(y_history) < 2:
        return np.nan # Not enough history

    # Naive error: mean(|t - (t-1)|)
    naive_errors = np.abs(np.diff(y_history.values))
    mae_naive = np.mean(naive_errors)

    if mae_naive == 0:
        return np.inf if mae_forecast > 0 else 0.0

    return mae_forecast / mae_naive

def plot_prediction(past_data, full_data, bolt_predictor, c2_predictions, known_covariates_future):
    """
    past_data: The data used for input (truncated)
    full_data: The original full data (containing the ground truth for evaluation)
    """
    print("\nGenerating Backtest Forecasts (Hiding last steps)...")
    PLOT_LENGTH= 200
    # 1. Generate Bolt predictions
    model_names = bolt_predictor.model_names()
    model_predictions = {}

    for model_name in model_names:
        print(f"Predicting with {model_name}...")
        model_predictions[model_name] = bolt_predictor.predict(past_data, known_covariates=known_covariates_future, model=model_name)

    for c2_name, c2_pred in c2_predictions.items():
        # Append to the list of names so the plotter loops over it
        model_names.append(c2_name)
        # Store the prediction dataframe
        model_predictions[c2_name] = c2_pred

    # 2. Add Chronos-2 predictions
    #for predictio
    #model_names.append("Chronos_2")
    #model_predictions["Chronos_2"] = c2_predictions

    # 3. Setup Plotting
    item_ids = sorted(full_data.reset_index()['item_id'].unique())
    num_plots = len(item_ids)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axs = plt.subplots(num_plots, 1, figsize=(15, 6 * num_plots), sharex=False)
    if num_plots == 1: axs = [axs]

    for i, item_id in enumerate(item_ids):
        ax = axs[i]

        # --- A. Prepare Data ---
        # Full history for plotting context
        history_context = full_data.loc[item_id].iloc[-(PLOT_LENGTH + PREDICTION_LENGTH):]

        # Ground Truth (The actual future we are trying to predict)
        # It corresponds to the last PREDICTION_LENGTH steps of the full data
        ground_truth_future = full_data.loc[item_id].iloc[-PREDICTION_LENGTH:]['pv_value']

        # History used for MASE denominator (everything BEFORE the prediction window)
        # Note: We take enough history to get a stable naive error, e.g., all available history or last 500 steps
        history_for_metric = full_data.loc[item_id].iloc[:-PREDICTION_LENGTH]['pv_value']

        # --- B. Plot Ground Truth ---
        ax.plot(history_context.index, history_context['pv_value'],
                label='Actual Ground Truth', color='black', linewidth=2, alpha=0.6)

        # Highlight the Forecast Window
        cutoff_date = ground_truth_future.index[0]
        ax.axvspan(cutoff_date, ground_truth_future.index[-1], color='gray', alpha=0.1, label="Forecast Window")

        # --- C. Plot Each Model & Calculate MASE ---
        for idx, model_name in enumerate(model_names):
            preds = model_predictions[model_name]
            forecast = preds.loc[item_id]

            # 1. Calculate MASE
            # Align forecast with ground truth (just to be safe, though indices should match)
            # We assume forecast index matches ground_truth_future index
            try:
                mase_score = calculate_item_mase(
                    y_true=ground_truth_future,
                    y_pred=forecast['mean'],
                    y_history=history_for_metric
                )
                mase_label = f"MASE: {mase_score:.3f}"
            except Exception as e:
                print(f"Error calc MASE for {model_name}: {e}")
                mase_label = "MASE: N/A"

            # 2. Prepare Label
            clean_name = model_name.split('/')[-1]
            label_text = f'{clean_name} | {mase_label}'
            color = colors[idx % len(colors)]

            # 3. Plot Mean
            ax.plot(forecast.index, forecast['mean'],
                    label=label_text, color=color, linewidth=2, linestyle='--')

            # 4. Plot Confidence Interval
            if '0.1' in forecast.columns and '0.9' in forecast.columns:
                ax.fill_between(
                    forecast.index,
                    forecast['0.1'],
                    forecast['0.9'],
                    color=color, alpha=0.15
                )

        # Formatting
        ax.set_title(f"Segment {item_id}: Backtest Performance", fontsize=14, fontweight='bold')
        ax.set_ylabel("PV Value", fontsize=10)
        ax.axvline(x=cutoff_date, color='red', linestyle=':', linewidth=1.5, label="Prediction Start")
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.show()

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
      "Chronos2FineTuned",
      "Chronos2ZeroShot",
      "WeightedEnsemble"
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
          preds.to_csv(f"forecast_{model_name}.csv")
      else:
          print(f"Warning: Model '{model_name}' not found in predictor.")

  return predictions_dict

  full_df = load_and_prepare(train_path)

print(f"Data shape: {full_df.shape}")
print(f"Unique Item IDs (Segments): {full_df.item_ids}")

# Fit models (Returns full trained data for context)
if not USE_CV2_SAVED_MODEL:
  train_data, bolt_predictor, c2_predictor = fit_model(full_df, use_saved_models=USE_CV2_SAVED_MODEL)
else:
  train_data, bolt_predictor, c2_predictor = fit_model(full_df, use_saved_models=USE_CV2_SAVED_MODEL)


# --- Generate Predictions ---
# 1. Create a "Past" dataset by slicing off the last PREDICTION_LENGTH steps
#    The model sees data up to (Now - 96 steps)
past_data = train_data.slice_by_timestep(None, -PREDICTION_LENGTH)
known_covariates_future = train_data[COVARIATE_COLUMNS]


# 2. Predict with Chronos-2 manually (Bolt is handled inside plot function loop)
print("Generating Chronos-2 predictions...")
cv2_model_predictions=chronos2prediction(past_data, known_covariates_future,known_covariates=known_covariates_future, c2_predictor=c2_predictor, use_saved_models=USE_CV2_SAVED_MODEL)

plot_prediction(past_data, train_data, bolt_predictor, cv2_model_predictions, known_covariates_future)


if not USE_CV2_SAVED_MODEL:
  cv2_model_names = c2_predictor.model_names()
  cv2_model_predictions = {}

  for model_name in cv2_model_names:
      print(f"Predicting with {model_name}...")
      cv2_model_predictions[model_name] = c2_predictor.predict(past_data, known_covariates=known_covariates_future, model=model_name)
else:
  cv2_model_predictions=chronos2prediction(past_data, known_covariates_future, use_saved_models=USE_CV2_SAVED_MODEL)

# 3. Plot with MASE
# Important: We pass 'train_data' (the full dataset) so we can compare predictions vs actuals
plot_prediction(past_data, train_data, bolt_predictor, cv2_model_predictions, known_covariates_future)
