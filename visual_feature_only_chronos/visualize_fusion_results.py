import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from transformers import AutoConfig, AutoImageProcessor, AutoModel
from peft import PeftModel
from visionFusion import VisionChronos2Model, MultimodalDataset, BASE_MODEL_NAME, VISION_MODEL_NAME, DATA_PATH, CONTEXT_LENGTH, PREDICTION_LENGTH, STRIDE
import os
from PIL import Image
import io

# --- Configuration ---
CHECKPOINT_DIR = "chronos_vision_fusion_checkpoints"

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [
        d for d in os.listdir(checkpoint_dir) 
        if d.startswith("checkpoint-epoch-") and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    latest = checkpoints[-1]
    print(f"Found Latest Checkpoint: {latest}")
    return os.path.join(checkpoint_dir, latest)

CHECKPOINT_PATH = get_latest_checkpoint(CHECKPOINT_DIR)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(checkpoint_path):
    print(f"Loading Base Config from {BASE_MODEL_NAME}...")
    config = AutoConfig.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model = VisionChronos2Model(config)
    
    print("Loading Base Chronos Weights...")
    base_model_file = AutoModel.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model.load_state_dict(base_model_file.state_dict(), strict=False)
    del base_model_file
    
    projector_path = os.path.join(checkpoint_path, "projector.pt")
    if os.path.exists(projector_path):
        print(f"Loading Projector Weights from {projector_path}...")
        projector_state_dict = torch.load(projector_path, map_location=DEVICE)
        model.vision_projector.load_state_dict(projector_state_dict)
    else:
        print("WARNING: Projector weights not found!")

    print(f"Loading LoRA Adapters from {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.to(DEVICE)
    model.eval()
    return model

# --- Metrics ---

def calculate_mape(y_true, y_pred, epsilon=1e-6):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def calculate_wmape(y_true, y_pred):
    """Weighted MAPE"""
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-6) * 100

def calculate_mase(y_true, y_pred, y_train, seasonality=1):
    """Mean Absolute Scaled Error"""
    # Naive forecast is y_train[:-seasonality] -> y_train[seasonality:] usually?
    # Or for simple TS: naive forecast is previous step.
    # Chronos context is the "Training Data" for this window.
    if len(y_train) < seasonality + 1:
        return np.nan
        
    # Naive MAE (In-sample 1-step ahead naive error)
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    mae_naive = np.mean(naive_errors) + 1e-6
    
    mae_model = np.mean(np.abs(y_true - y_pred))
    
    return mae_model / mae_naive

# --- Evaluation ---

def evaluate_last_window(model, df, img_processor, series_id):
    # Filter for Series
    group = df[df["series_id"] == series_id].sort_values("time") # Ensure sort
    
    # Get Last Window
    # Size = Context + Pred
    window_size = CONTEXT_LENGTH + PREDICTION_LENGTH
    if len(group) < window_size:
        print(f"Series {series_id} too short!")
        return None
        
    last_window = group.iloc[-window_size:]
    
    # Prepare Input
    series_values = last_window["pv"].values.astype(np.float32)
    context = series_values[:CONTEXT_LENGTH]
    target = series_values[CONTEXT_LENGTH:]
    
    # Image (Last in context)
    img_data = last_window["image"].iloc[CONTEXT_LENGTH - 1] 
    if isinstance(img_data, dict) and 'bytes' in img_data:
        img = Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")
    else:
        img = Image.new('RGB', (224, 224))
    
    image_tensor = img_processor(img, return_tensors="pt")["pixel_values"].to(DEVICE) # (1, 3, H, W)
    context_tensor = torch.tensor(context).to(DEVICE).unsqueeze(0) # (1, Context)
    
    # Predict
    with torch.no_grad():
        fc_quantiles = model.predict(
            context=context_tensor,
            image_tensors=image_tensor,
            prediction_length=PREDICTION_LENGTH
        )
        # Debug Stats
        print(f"  [Debug] Forecast Shape: {fc_quantiles.shape}")
        print(f"  [Debug] Forecast Mean: {fc_quantiles.mean().item():.4f}, Std: {fc_quantiles.std().item():.4f}")
        print(f"  [Debug] Target Mean: {target.mean():.4f}, Std: {target.std():.4f}")
        
        fc_quantiles = fc_quantiles.cpu().numpy() # (1, Quantiles, Horizon)

    # Decode Forecast
    num_quantiles = fc_quantiles.shape[1]
    if num_quantiles > 1:
        median_idx = 4 if num_quantiles == 9 else num_quantiles // 2
        forecast_median = fc_quantiles[0, median_idx, :]
    else:
        forecast_median = fc_quantiles[0, 0, :]
        
    # Calculate Metrics
    mape = calculate_mape(target, forecast_median)
    wmape = calculate_wmape(target, forecast_median)
    # Use context as training history for MASE, assume seasonality=96 (daily at 15min? wait, data is 30min? or 15min?)
    # SKIPP'D is 15min usually? 24h = 96 steps. 
    # Actually Dataset frequency check showed 30min? No, 15min (96 steps per day).
    mase = calculate_mase(target, forecast_median, context, seasonality=96)
    
    results = {
        "series_id": series_id,
        "mape": mape,
        "wmape": wmape,
        "mase": mase,
        "context": context,
        "target": target,
        "forecast": forecast_median,
        "quantiles": fc_quantiles
    }
    
    return results

def plot_results(res):
    series_id = res["series_id"]
    context = res["context"]
    target = res["target"]
    forecast = res["forecast"]
    quantiles = res["quantiles"]
    
    plt.figure(figsize=(15, 6))
    
    # History (Zoom in to last 200 steps for clarity)
    zoom_len = 200
    plt.plot(range(len(context)-zoom_len, len(context)), context[-zoom_len:], label="History (Zoom)", color="black", alpha=0.5)
    
    # Forecast
    time_future = range(len(context), len(context) + len(target))
    plt.plot(time_future, target, label="Ground Truth", color="green", linewidth=2)
    plt.plot(time_future, forecast, label="Forecast (Median)", color="blue", linewidth=2)
    
    # CI
    if quantiles.shape[1] > 1:
        low = quantiles[0, 1, :]
        high = quantiles[0, 7, :]
        plt.fill_between(time_future, low, high, color="blue", alpha=0.2, label="80% CI")
        
    plt.title(f"Series: {series_id} | wMAPE: {res['wmape']:.2f}% | MASE: {res['mase']:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = f"eval_last_window_{series_id}.png"
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

def main():
    model = load_model(CHECKPOINT_PATH)
    
    print("Loading DataFrame...")
    df = pd.read_parquet(DATA_PATH)
    img_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_NAME)
    
    # Identify Series
    unique_series = df["series_id"].unique()
    print(f"Found {len(unique_series)} Series: {unique_series}")
    
    avg_wmape = 0
    
    for sid in unique_series:
        print(f"\nEvaluating Series: {sid}...")
        res = evaluate_last_window(model, df, img_processor, sid)
        if res:
            print(f"  MAPE:  {res['mape']:.2f}%")
            print(f"  wMAPE: {res['wmape']:.2f}%")
            print(f"  MASE:  {res['mase']:.4f}")
            plot_results(res)
            avg_wmape += res['wmape']
            
    print(f"\n--- Average wMAPE: {avg_wmape / len(unique_series):.2f}% ---")

if __name__ == "__main__":
    main()
