import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from chronos import ChronosPipeline, BaseChronosPipeline
from peft import PeftModel

# Try importing Chronos2Pipeline if available
try:
    from chronos.chronos2.pipeline import Chronos2Pipeline
except ImportError:
    Chronos2Pipeline = None

# ==========================================
# Configuration
# ==========================================
DATA_PATH = "/content/drive/MyDrive/FM_project/dataset/skippd_train_5min_no_images.parquet"
MODELS_TO_RUN = [ "amazon/chronos-t5-small", "amazon/chronos-bolt-small","amazon/chronos-bolt-base", "amazon/chronos-2"]
CHECKPOINT_ROOT = "/content/MyDrive/FM_project/chronos_finetuning_checkpoints"

CONTEXT_LENGTH = 600
PREDICTION_LENGTH = 60
PLOT_CONTEXT_LEN = 100 # How much context to show in plot

# ==========================================
# Helper Functions
# ==========================================

def get_pipeline_class(model_name):
    if "chronos-2" in model_name:
        if Chronos2Pipeline is None:
             raise ImportError("Chronos2Pipeline not found. Ensure chronos-forecasting is installed correctly.")
        return Chronos2Pipeline
    elif "bolt" in model_name:
        return BaseChronosPipeline
    return ChronosPipeline

def get_correct_model_input_size(model_name, context_list):
    """
    Adjusts input shape based on model type.
    """
    if "chronos-2" in model_name:
        # Chronos-2 often expects (batch, channels, time) or similar, depending on version
        # Based on baseline script: (1, 1, 512)
        if hasattr(context_list, "unsqueeze"):
             return context_list.unsqueeze(0).unsqueeze(0)
    # Default for T5/Bolt: (1, 512)
    return context_list.unsqueeze(0)

def select_best_adapter_folder(model_name):
    """
    Finds the latest fold's 'final_adapter' for the given model.
    """
    safe_name = model_name.replace("/", "_").replace("-", "_")
    checkpoint_base = os.path.join(CHECKPOINT_ROOT, safe_name)
    
    if not os.path.exists(checkpoint_base):
         print(f"Warning: Checkpoint directory not found: {checkpoint_base}")
         return None

    fold_dirs = [
        d for d in os.listdir(checkpoint_base)
        if d.startswith("fold_") and os.path.isdir(os.path.join(checkpoint_base, d))
    ]

    if fold_dirs:
        # Sort numerically by fold number
        fold_dirs.sort(key=lambda x: int(x.split('_')[-1]))
        latest_fold = fold_dirs[-1]
        adapter_path = os.path.join(checkpoint_base, latest_fold, "final_adapter")
        print(f"Selected adapter for {model_name}: {adapter_path}")
        return adapter_path
    else:
        print(f"No 'fold_*' directories found in {checkpoint_base}")
        return None

def get_forecast(pipeline, model_input, prediction_length):
    forecast = pipeline.predict(
        model_input,
        prediction_length=prediction_length,
    )
    # Convert to numpy and fix shape
    forecast_numpy = forecast[0].numpy()
    
    # Handle varying output shapes (batch, samples, time) or (samples, time)
    if forecast_numpy.ndim == 3 and forecast_numpy.shape[0] == 1:
        forecast_numpy = forecast_numpy[0]
    if forecast_numpy.ndim == 3 and forecast_numpy.shape[-1] == 1:
        forecast_numpy = forecast_numpy[..., 0]
        
    return forecast_numpy

# ==========================================
# Main Execution
# ==========================================

def main():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_parquet(DATA_PATH)
        
        if 'pv' not in df.columns:
            raise ValueError("Column 'pv' not found in dataset")
        
        full_series = df["pv"].values
        
        # Dataset Split (Visualizing the end of the series, similar to baseline)
        total_len = len(full_series)
        split_point = total_len - PREDICTION_LENGTH
        context_start = split_point - CONTEXT_LENGTH
        
        # Verify indices
        if context_start < 0:
            raise ValueError("Time series is too short for the requested context and prediction length.")

        # Extract tensors
        context_tensor = full_series[context_start : split_point]
        ground_truth = full_series[split_point : total_len]
        
        context_list = torch.tensor(context_tensor, dtype=torch.float32)
        
        print(f"Context range: {context_start} to {split_point}")
        print(f"Prediction range: {split_point} to {total_len}")

        # Prepare X-axis
        if 'time' in df.columns:
            time_series = pd.to_datetime(df['time'].values)
            x_context = time_series[context_start:split_point]
            x_future = time_series[split_point:total_len]
        else:
            x_context = np.arange(context_start, split_point)
            x_future = np.arange(split_point, total_len)

        # Plot Setup
        fig, axes = plt.subplots(nrows=len(MODELS_TO_RUN), ncols=1, figsize=(12, 6 * len(MODELS_TO_RUN)), sharex=False)
        if len(MODELS_TO_RUN) == 1: axes = [axes]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for i, model_name in enumerate(MODELS_TO_RUN):
            print(f"\n--- Processing Model: {model_name} ---")
            ax = axes[i]
            pipeline_class = get_pipeline_class(model_name)
            
            # ---------------------------
            # 1. Original Model Response
            # ---------------------------
            print(" Generating Original Forecast...")
            pipeline_orig = pipeline_class.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.bfloat16,
            )
            
            model_input = get_correct_model_input_size(model_name, context_list)
            forecast_orig = get_forecast(pipeline_orig, model_input, PREDICTION_LENGTH)
            
            # Clean up to free memory
            del pipeline_orig
            torch.cuda.empty_cache()

            # ---------------------------
            # 2. Fine-Tuned Model Response
            # ---------------------------
            print(" Generating Fine-Tuned Forecast...")
            adapter_path = select_best_adapter_folder(model_name)
            
            if adapter_path and os.path.exists(adapter_path):
                # Reload base model
                pipeline_ft = pipeline_class.from_pretrained(
                    model_name,
                    device_map=device,
                    torch_dtype=torch.bfloat16,
                )
                
                # Apply LoRA Adapter
                # Structure depends on model type
                if "bolt" in model_name:
                    target_model = pipeline_ft.model
                elif "chronos-2" in model_name:
                    target_model = pipeline_ft.model
                else: 
                    # T5 / Original Chronos
                    target_model = pipeline_ft.model.model

                # Load adapter
                # We need to wrap the internal model with PeftModel
                # Note: get_peft_model returns a PeftModel. load_adapter might be cleaner if supported, 
                # but PeftModel.from_pretrained is standard for inference.
                
                # Careful: The pipeline holds a reference to the model. We need to replace it or modify it in place.
                # PeftModel.from_pretrained wraps the base model.
                peft_model = PeftModel.from_pretrained(target_model, adapter_path)
                
                # Re-assign back to pipeline if necessary
                if "bolt" in model_name:
                    pipeline_ft.model = peft_model
                elif "chronos-2" in model_name:
                    pipeline_ft.model = peft_model
                else:
                    pipeline_ft.model.model = peft_model

                forecast_ft = get_forecast(pipeline_ft, model_input, PREDICTION_LENGTH)
                
                del pipeline_ft
                torch.cuda.empty_cache()
                has_finetuned = True
            else:
                print(f"Skipping fine-tuned forecast for {model_name} (Adapter not found).")
                has_finetuned = False
                forecast_ft = None

            # ---------------------------
            # 3. Visualization
            # ---------------------------
            median_orig = np.median(forecast_orig, axis=0)
            low_orig = np.quantile(forecast_orig, 0.1, axis=0) # 10th percentile
            high_orig = np.quantile(forecast_orig, 0.9, axis=0) # 90th percentile

            # Plot Context
            ax.plot(x_context[-PLOT_CONTEXT_LEN:], context_tensor[-PLOT_CONTEXT_LEN:], label="Context", color="black", alpha=0.6)
            
            # Plot Ground Truth
            ax.plot(x_future, ground_truth, label="Ground Truth", color="green", linewidth=2.5)
            
            # Plot Original
            ax.plot(x_future, median_orig, label="Original Model (Median)", color="blue", linestyle="--")
            ax.fill_between(x_future, low_orig, high_orig, color="blue", alpha=0.1, label="Original 80% CI")

            # Plot Fine-Tuned
            if has_finetuned and forecast_ft is not None:
                median_ft = np.median(forecast_ft, axis=0)
                low_ft = np.quantile(forecast_ft, 0.1, axis=0)
                high_ft = np.quantile(forecast_ft, 0.9, axis=0)
                
                ax.plot(x_future, median_ft, label="Fine-Tuned (Median)", color="red", linestyle="--")
                ax.fill_between(x_future, low_ft, high_ft, color="red", alpha=0.1, label="Fine-Tuned 80% CI")

            ax.set_title(f"Forecast Comparison: {model_name}")
            
            if 'time' in df.columns:
                date_format = mdates.DateFormatter('%d-%m %H:%M')
                ax.xaxis.set_major_formatter(date_format)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_plot = 'chronos_finetune_comparison.png'
        plt.savefig(output_plot)
        print(f"\nSaved comparison plot to '{output_plot}'")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
