import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from chronos import ChronosPipeline, BaseChronosPipeline
try:
    from chronos.chronos2.pipeline import Chronos2Pipeline
except ImportError:
    Chronos2Pipeline = None

from peft import PeftModel

# --- Configuration ---
DATA_PATH = "/content/drive/MyDrive/FM_project/dataset/skippd_train_no_images.parquet"
MODEL_NAME = "amazon/chronos-2" # Change this as needed
# Placeholder - Logic to find best adapter automatically or set manually
# ADAPTER_PATH = "/content/chronos_finetuning_checkpoints/amazon_chronos_2/fold_1/final_adapter" 
CONTEXT_LENGTH = 600
PREDICTION_LENGTH = 60
NUM_SAMPLES = 4

def get_best_adapter(model_name):
    # Same logic as LoRA.py to find the latest fold's adapter
    safe_name = model_name.replace("/", "_").replace("-", "_")
    checkpoint_base = f"/content/chronos_finetuning_checkpoints/{safe_name}"
    
    if not os.path.exists(checkpoint_base):
        print(f"Warning: Checkpoint directory {checkpoint_base} not found. Using Base model only if not found.")
        return None

    fold_dirs = [
        d for d in os.listdir(checkpoint_base)
        if d.startswith("fold_") and os.path.isdir(os.path.join(checkpoint_base, d))
    ]

    if fold_dirs:
        fold_dirs.sort(key=lambda x: int(x.split('_')[-1]))
        latest_fold = fold_dirs[-1]
        adapter_path = os.path.join(checkpoint_base, latest_fold, "final_adapter")
        print(f"Found Adapter: {adapter_path}")
        return adapter_path
    
    return None

def load_pipeline(model_name):
    if "chronos-2" in model_name:
        cls = Chronos2Pipeline
    elif "bolt" in model_name:
        cls = BaseChronosPipeline
    else:
        cls = ChronosPipeline
        
    return cls.from_pretrained(
        model_name,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )

def main():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_parquet(DATA_PATH)
        full_series = df['pv'].values
        
        # Prepare Data Indices for Visualization
        # We want 4 distinct windows from the validation/test part (end of series)
        total_len = len(full_series)
        test_size = int(total_len * 0.2) # Last 20%
        test_start_idx = total_len - test_size
        
        # Select 4 random start points in validity range
        # Valid start point must allow (start -> start+context+prediction) within available data
        valid_range_end = total_len - (CONTEXT_LENGTH + PREDICTION_LENGTH)
        
        if valid_range_end <= test_start_idx:
            # Fallback if series is too short, just pick from available range
            indices = np.linspace(0, valid_range_end, NUM_SAMPLES, dtype=int)
        else:
             indices = np.linspace(test_start_idx, valid_range_end, NUM_SAMPLES, dtype=int)

        # 1. Load Base Model
        print(f"Loading Base Model: {MODEL_NAME}")
        pipeline = load_pipeline(MODEL_NAME)
        
        # 2. Get predictions for Base Model
        print("Generating Base Model Forecasts...")
        base_forecasts = []
        contexts = []
        ground_truths = []
        
        for idx in indices:
            window = full_series[idx : idx + CONTEXT_LENGTH + PREDICTION_LENGTH]
            context = window[:CONTEXT_LENGTH]
            gt = window[CONTEXT_LENGTH:]
            
            contexts.append(context)
            ground_truths.append(gt)
            
            context_tensor = torch.tensor(context, dtype=torch.float32)
            # Chronos 2/Bolt expects raw tensor usually, T5 tokenizes inside.
            # Pipeline.predict handles the difference mostly, but let's be consistent with input.
            # Predict expects (batch, time) or list of (time)
            
            f = pipeline.predict(
                torch.tensor([context]), # (1, time)
                prediction_length=PREDICTION_LENGTH
            )
            base_forecasts.append(f)

        # 3. Load Fine-Tuned Model
        adapter_path = get_best_adapter(MODEL_NAME)
        finetuned_forecasts = []
        
        if adapter_path:
            print(f"Loading Adapter from {adapter_path}...")
            # We need to apply the adapter to the underlying model
            
            if "t5" in MODEL_NAME:
                model_to_patch = pipeline.model.model
            else:
                model_to_patch = pipeline.model # Bolt/Chronos2
                
            # Load PEFT adapter
            peft_model = PeftModel.from_pretrained(model_to_patch, adapter_path)
            
            # Replace the model in pipeline with the peft model
            # For T5: pipeline.model.model = peft_model
            # For Others: pipeline.model = peft_model
            if "t5" in MODEL_NAME:
                pipeline.model.model = peft_model
            else:
                pipeline.model = peft_model
                
            print("Generating Fine-Tuned Model Forecasts...")
            for idx, context in zip(indices, contexts):
                 f = pipeline.predict(
                    torch.tensor([context]),
                    prediction_length=PREDICTION_LENGTH
                )
                 finetuned_forecasts.append(f)
        else:
            print("No adapter found. Skipping Fine-Tuned forecasts.")
            finetuned_forecasts = [None] * NUM_SAMPLES

        # 4. Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            context = contexts[i]
            gt = ground_truths[i]
            base_f = base_forecasts[i].numpy() # (1, num_samples, pred_len)
            ft_f = finetuned_forecasts[i]
            
            # X-axis indices
            x_context = np.arange(len(context))
            x_future = np.arange(len(context), len(context) + len(gt))
            
            # Plot Context (Last 100 points for clarity)
            plot_lookback = 150
            ax.plot(x_context[-plot_lookback:], context[-plot_lookback:], color="black", alpha=0.6, label="Context")
            
            # Plot GT
            ax.plot(x_future, gt, color="green", linewidth=2, label="Ground Truth")
            
            # Plot Base
            # Median
            base_median = np.median(base_f[0], axis=0)
            ax.plot(x_future, base_median, color="blue", linestyle="--", label="Base (Median)")
            
            # Plot Fine-Tuned
            if ft_f is not None:
                ft_f_np = ft_f.numpy()
                ft_median = np.median(ft_f_np[0], axis=0)
                ax.plot(x_future, ft_median, color="red", label="Fine-Tuned (Median)")
                
                # Validation Loss usually on 'future_target', visualizing improvement
            
            ax.set_title(f"Sample {i+1} (idx={indices[i]})")
            ax.grid(True, alpha=0.3)
            
            if i == 0: # Legend only on first
                ax.legend()
        
        plt.tight_layout()
        save_path = "fine_tune/comparison_plot.png"
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
