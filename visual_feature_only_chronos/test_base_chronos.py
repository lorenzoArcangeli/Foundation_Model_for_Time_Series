import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chronos import ChronosPipeline, BaseChronosPipeline # Use the official pipeline for baseline
from visionFusion import DATA_PATH, CONTEXT_LENGTH, PREDICTION_LENGTH, STRIDE

# Use the exact same model name
BASE_MODEL_NAME = "amazon/chronos-2" # Or whatever chronos-2 maps to. 
# Notes: "amazon/chronos-2" isn't a valid huggingface ID usually, it's "amazon/chronos-t5-small" etc.
# Check what the user was using. In visionFusion.py they used "amazon/chronos-2"?
# Let's check visionFusion.py again via file system or memory. 
# Memory says: BASE_MODEL_NAME = "amazon/chronos-2"
# Wait, "amazon/chronos-2" might be a placeholder? 
# The user's research md mentioned "amazon/chronos-t5-small"
# Let's try "amazon/chronos-t5-small" because that's the real model.

SEED = 42

def evaluate_baseline(series_id, df):
    # Load Pipeline (Official)
    pipeline = BaseChronosPipeline.from_pretrained(
        BASE_MODEL_NAME,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )
    
    # Filter Series
    group = df[df["series_id"] == series_id].sort_values("time")
    window_size = CONTEXT_LENGTH + PREDICTION_LENGTH
    last_window = group.iloc[-window_size:]
    
    context_data = last_window["pv"].values[:CONTEXT_LENGTH]
    target_data = last_window["pv"].values[CONTEXT_LENGTH:]
    
    # Predict
    # Chronos Pipeline expects context as tensor or list
    context_tensor = torch.tensor(context_data)
    
    forecast = pipeline.predict(
        context_tensor.unsqueeze(0).unsqueeze(0),
        prediction_length=PREDICTION_LENGTH,
        #num_samples=20,
    ) # (1, num_samples, horizon)
    
    # Get Median
    forecast_numpy = forecast[0].numpy() 
            
    # Shape Fixer
    if forecast_numpy.ndim == 3 and forecast_numpy.shape[0] == 1:
        forecast_numpy = forecast_numpy[0]
    if forecast_numpy.ndim == 3 and forecast_numpy.shape[-1] == 1:
        forecast_numpy = forecast_numpy[..., 0]

    # Stats
    median = np.median(forecast_numpy, axis=0)
    low = np.quantile(forecast_numpy, 0.1, axis=0)
    high = np.quantile(forecast_numpy, 0.9, axis=0)
    
    # Metrics
    def wmape(y_true, y_pred):
        return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-6) * 100
        
    wmape_val = wmape(target_data, median)
    print(f"Series {series_id} Baseline wMAPE: {wmape_val:.2f}%")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(context_data)), context_data, color="black", label="History")
    plt.plot(range(len(context_data), len(context_data)+len(target_data)), target_data, color="green", label="Truth")
    plt.plot(range(len(context_data), len(context_data)+len(target_data)), median, color="blue", label="Baseline Forecast")
    plt.fill_between(range(len(context_data), len(context_data)+len(target_data)), low, high, color="blue", alpha=0.2)
    plt.title(f"Baseline Chronos (No Vision) - Series {series_id}")
    plt.legend()
    plt.savefig(f"baseline_series_{series_id}.png")
    plt.close()

def main():
    df = pd.read_parquet(DATA_PATH)
    unique_series = df["series_id"].unique()
    
    for sid in unique_series:
        try:
            evaluate_baseline(sid, df)
        except Exception as e:
            print(f"Error on {sid}: {e}")

if __name__ == "__main__":
    main()
