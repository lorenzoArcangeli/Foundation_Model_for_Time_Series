import os
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, default_collate
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union

# Import Chronos components
from chronos import ChronosPipeline, BaseChronosPipeline
try:
    from chronos.chronos2.pipeline import Chronos2Pipeline
except ImportError:
    Chronos2Pipeline = None

# --- Configuration ---

@dataclass
class TrainingConfig:
    data_path: str = "/content/drive/MyDrive/FM_project/dataset/skippd_train_no_images.parquet"
    model_name: str = "amazon/chronos-2"  # Options: "amazon/chronos-t5-small", "amazon/chronos-bolt-small", "amazon/chronos-2"
    context_length: int = 600
    prediction_length: int = 60
    n_splits: int = 2
    output_dir: str = "chronos_finetuning_checkpoints"
    batch_size: int = 32
    num_epochs: int = 1
    learning_rate: float = 1e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    covariate_cols: List[str] = field(default_factory=lambda: ['time_hour_sin', 'time_hour_cos', 'time_dayofyear_sin']) # Add more as needed, e.g. sky features
    
    @property
    def safe_model_name(self):
        return self.model_name.replace("/", "_").replace("-", "_")

# Global config instance
CONFIG = TrainingConfig()

# --- Utility Functions ---

def select_best_adapter_folder(model_name: str) -> str:
    safe_name = model_name.replace("/", "_").replace("-", "_")
    checkpoint_base = f"/content/chronos_finetuning_checkpoints/{safe_name}"
    
    if not os.path.exists(checkpoint_base):
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_base} does not exist.")

    fold_dirs = [
        d for d in os.listdir(checkpoint_base)
        if d.startswith("fold_") and os.path.isdir(os.path.join(checkpoint_base, d))
    ]

    if fold_dirs:
        # Sort by fold number
        fold_dirs.sort(key=lambda x: int(x.split('_')[-1]))
        latest_fold = fold_dirs[-1]
        adapter_path = os.path.join(checkpoint_base, latest_fold, "final_adapter")
        print(f"Selected adapter: {adapter_path}")
        return adapter_path
    else:
        raise FileNotFoundError(f"No 'fold_*' directories found in {checkpoint_base}")

def get_patch_size(model) -> int:
    """Safely retrieves the output patch size from the model config."""
    if hasattr(model, "chronos_config"):
        return getattr(model.chronos_config, "output_patch_size", 16)
    elif hasattr(model, "config") and hasattr(model.config, "chronos_config"):
        return getattr(model.config.chronos_config, "output_patch_size", 16)
    return 16  # Fallback

def chronos_collator(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collator to handle 'num_output_patches' scalar."""
    num_output_patches = None
    clean_batch = []
    
    for item in batch:
        item_copy = item.copy()
        if "num_output_patches" in item_copy:
            val = item_copy.pop("num_output_patches")
            if num_output_patches is None:
                num_output_patches = val
        clean_batch.append(item_copy)
    
    collated = default_collate(clean_batch)
    
    if num_output_patches is not None:
        collated["num_output_patches"] = int(num_output_patches)
        
    return collated

def calculate_mase(y_true, y_pred, y_history):
    """
    Mean Absolute Scaled Error (MASE).
    y_true: Actual values (future).
    y_pred: Forecast values.
    y_history: Historical values (for naive forecast scale).
    """
    # Naive forecast errors (step 1 difference)
    n = len(y_history)
    d = np.abs(np.diff(y_history)).mean()
    
    # Avoid division by zero
    if d == 0:
        return np.inf
    
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

def calculate_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE).
    """
    # Filter out zeros to avoid infinite errors or handle them gracefully
    # Here we mask where y_true is 0.
    mask = y_true != 0
    if not np.any(mask):
        return np.inf
        
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# --- Dataset Handler ---

class BaseChronosDataset(Dataset):
    def __init__(self, time_series_data: List[np.ndarray], pipeline, context_length, prediction_length, model_type, covariates_data: List[np.ndarray] = None):
        """
        time_series_data: List[np.ndarray]
                          A list where each element is a full time series (e.g. from one series_id).
        covariates_data: List[np.ndarray], optional
                         A list where each element is the corresponding covariates 2D array (Time, N_Covariates).
                         Must match time_series_data in length and order.
        """
        self.pipeline = pipeline
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.model_type = model_type
        
        # Prepare sliding windows
        self.samples = []
        total_window = self.context_length + self.prediction_length
        
        # Ensure input is a list
        if not isinstance(time_series_data, list):
            time_series_data = [time_series_data]
            if covariates_data is not None:
                covariates_data = [covariates_data]

        for idx, series in enumerate(time_series_data):
            if isinstance(series, torch.Tensor):
                series = series.numpy()
            
            # Helper for covariates
            series_covariates = None
            if covariates_data is not None:
                series_covariates = covariates_data[idx]
                if len(series_covariates) != len(series):
                    print(f"Warning: Covariates length ({len(series_covariates)}) mismatch series length ({len(series)}). Trimming to min.")
                    min_len = min(len(series), len(series_covariates))
                    series = series[:min_len]
                    series_covariates = series_covariates[:min_len]

            # Skip if series is too short
            if len(series) < total_window:
                continue
                
            # Create samples for this specific series
            for i in range(len(series) - total_window + 1):
                window = series[i : i + total_window]
                
                sample = {"window": window}
                
                if series_covariates is not None:
                     cov_window = series_covariates[i : i + total_window]
                     sample["covariates"] = cov_window
                
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__")


class ChronosT5Dataset(BaseChronosDataset):
    def __init__(self, time_series, pipeline, context_length, prediction_length, covariates=None):
        # T5 Model usually doesn't support covariates in the standard pipeline easily, ignoring for now or TBD
        super().__init__(time_series, pipeline, context_length, prediction_length, "t5", covariates)
        self.tokenizer = pipeline.tokenizer
        if covariates is not None:
             print("Warning: ChronosT5Dataset does not currently implement covariate support. They will be ignored.")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        window = sample["window"]
        
        context = torch.tensor(window[:self.context_length])
        target = torch.tensor(window[self.context_length:])
        
        context_batch = context.unsqueeze(0)
        target_batch = target.unsqueeze(0)
        
        input_ids, attention_mask, tokenizer_state = self.tokenizer.context_input_transform(context_batch)
        labels, labels_mask = self.tokenizer.label_input_transform(target_batch, tokenizer_state)
        
        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0)
        }


class ChronosBoltDataset(BaseChronosDataset):
    def __init__(self, time_series, pipeline, context_length, prediction_length, covariates=None):
        super().__init__(time_series, pipeline, context_length, prediction_length, "bolt", covariates)
        if covariates is not None:
             print("Warning: ChronosBoltDataset currently mostly univariate in this script. Covariates ignored.")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        window = sample["window"]
        
        # Bolt expects raw tensors and boolean masks
        # 1. Define Raw Tensors (containing potential NaNs)
        raw_context = torch.tensor(window[:self.context_length], dtype=torch.float32)
        raw_target = torch.tensor(window[self.context_length:], dtype=torch.float32)

        # 2. Create Masks: True (1) for real data, False (0) for NaNs
        context_mask = ~torch.isnan(raw_context)
        target_mask = ~torch.isnan(raw_target)

        # 3. Calculate Scale (Mean Abs of Context) to Normalize
        valid_context = raw_context[context_mask]
        if valid_context.numel() > 0:
            scale = torch.mean(torch.abs(valid_context))
            if scale == 0:
                scale = torch.tensor(1.0)
        else:
            scale = torch.tensor(1.0)

        # 4. Fill NaNs & Normalize
        context = torch.nan_to_num(raw_context, nan=0.0) / scale
        target = torch.nan_to_num(raw_target, nan=0.0) / scale

        return {
            "context": context,
            "target": target,
            "mask": context_mask,
            "target_mask": target_mask,
            "scale": scale 
        }


class Chronos2Dataset(BaseChronosDataset):
    def __init__(self, time_series, pipeline, context_length, prediction_length, covariates=None):
        super().__init__(time_series, pipeline, context_length, prediction_length, "chronos-2", covariates)
        self.output_patch_size = get_patch_size(pipeline.model)
        self.has_covariates = covariates is not None

    def __getitem__(self, idx):
        sample = self.samples[idx]
        window = sample["window"]
        
        # Chronos-2 uses 'future_target' and needs 'num_output_patches'
        # 1. Define Raw Tensors
        raw_context = torch.tensor(window[:self.context_length], dtype=torch.float32)
        raw_target = torch.tensor(window[self.context_length:], dtype=torch.float32)

        # 2. Create Masks
        context_mask = ~torch.isnan(raw_context)
        target_mask = ~torch.isnan(raw_target)

        # 3. Calculate Scale
        valid_context = raw_context[context_mask]
        if valid_context.numel() > 0:
            scale = torch.mean(torch.abs(valid_context))
            if scale == 0:
                scale = torch.tensor(1.0)
        else:
            scale = torch.tensor(1.0)

        # 4. Fill NaNs & Normalize
        context = torch.nan_to_num(raw_context, nan=0.0) / scale
        target = torch.nan_to_num(raw_target, nan=0.0) / scale

        # Calculate patches required for relevant lengths
        # 1. Prediction length (target)
        target_patches = math.ceil(self.prediction_length / self.output_patch_size)
        
        # 2. Future covariates length (if present)
        # Note: We flattened covariates to (L * V), so we need enough patches to cover this.
        # The error `found: 180 > 4 * 16` implies we need capacity for the flattened tensor.
        cov_patches = 0
        if self.has_covariates:
             # covariates is (TotalLen, NumContext). future is (PredLen, NumContext). Flattened is PredLen*NumContext.
             num_future_cov_tokens = self.prediction_length * self.time_series.shape[1] if len(self.time_series.shape) > 1 else 0
             # Actually, simpler: we know the flattened size will be calculated below.
             # We perform the split and flatten FIRST, then calculate num_patches.
             pass 

        # Let's perform the split first to get actual tensor sizes
            "context_mask": context_mask,
            "future_target_mask": target_mask,
            "num_output_patches": num_patches,
            # "scale": scale # Removed scale as it causes TypeError in model.forward()
        }
        
        # 5. Handle Covariates if present
        # Chronos 2 expects covariates to be passed. 
        # CAUTION: We assume the model was trained/configured to accept the dimensions.
        # Typically Chronos-2 handles 'past_covariates' and 'known_future_covariates'.
        # Since we have the full window, we split it.
        
        if "covariates" in sample:
            cov_window = sample["covariates"] # Shape: (TotalLen, NumFeatures)
            
            # Split into context (past) and prediction (future)
            # Note: The model signature only accepts `future_covariates`.
            # We assume this corresponds to the prediction horizon.
            
            # past_cov = torch.tensor(cov_window[:self.context_length], dtype=torch.float32) # Not used by model forward
            future_cov = torch.tensor(cov_window[self.context_length:], dtype=torch.float32)
            
            # Fill NaNs in covariates with 0
            # past_cov = torch.nan_to_num(past_cov, nan=0.0)
            future_cov = torch.nan_to_num(future_cov, nan=0.0)

            # Flatten to (FutureLength * NumFeatures) to match (B, L) expectation
            # This is a guess based on the error message rejecting 3D input.
            # If the model really expects (B, T), then we might need to select 1 feature or use a different approach.
            future_cov = future_cov.reshape(-1)
            
            # Add to batch
            # Matching signature: (..., future_covariates=..., future_covariates_mask=...)
            
            # batch_item["past_covariates"] = past_cov # Removed as per signature
            batch_item["future_covariates"] = future_cov
            batch_item["future_covariates_mask"] = torch.ones_like(future_cov, dtype=torch.bool)
            
        return batch_item

class DatasetFactory:
    @staticmethod
    def get_dataset(model_type: str, time_series, pipeline, context_length, prediction_length, covariates=None) -> BaseChronosDataset:
        if model_type == "t5":
            return ChronosT5Dataset(time_series, pipeline, context_length, prediction_length, covariates)
        elif model_type == "bolt":
            return ChronosBoltDataset(time_series, pipeline, context_length, prediction_length, covariates)
        elif model_type == "chronos-2":
            return Chronos2Dataset(time_series, pipeline, context_length, prediction_length, covariates)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

# --- Model Manager ---

class ChronosModelManager:
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.model_type = self._determine_model_type()
        
    def _determine_model_type(self) -> str:
        if "bolt" in self.model_name.lower():
            return "bolt"
        elif "chronos-2" in self.model_name.lower() or "710m-2" in self.model_name.lower():
            return "chronos-2"
        else:
            return "t5"

    def load_pipeline(self):
        print(f"Loading {self.model_type} pipeline for {self.model_name}...")
        
        if self.model_type == "bolt":
            self.pipeline = BaseChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
            )
        elif self.model_type == "chronos-2":
            if Chronos2Pipeline is None:
                raise ImportError("Chronos2Pipeline not found. Ensure chronos-forecasting is installed.")
            self.pipeline = Chronos2Pipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
            )
        else: # T5
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
            )
        return self.pipeline

    def get_peft_model(self, lora_config: LoraConfig):
        # Determine the model object to wrap
        if self.model_type == "t5":
            model_to_train = self.pipeline.model.model
        else:
            # Bolt and Chronos-2 expose the model directly suitable for PEFT
            model_to_train = self.pipeline.model
            
        peft_model = get_peft_model(model_to_train, lora_config)
        peft_model.print_trainable_parameters()
        return peft_model
    
    def get_lora_config(self) -> LoraConfig:
        # Determine TaskType
        # Bolt/Chronos-2 don't fit standard SEQ_2_SEQ_LM task type in PEFT easily or cause errors if used
        # T5 uses SEQ_2_SEQ_LM
        task_type = TaskType.SEQ_2_SEQ_LM if self.model_type == "t5" else None
        
        # Target modules can vary, but standard "q", "v" etc usually work for transformers
        return LoraConfig(
            r=CONFIG.lora_r,
            lora_alpha=CONFIG.lora_alpha,
            target_modules=["q", "v", "k", "o"],
            lora_dropout=CONFIG.lora_dropout,
            bias="none",
            task_type=task_type
        )
        
    def get_label_names(self) -> List[str]:
        if self.model_type == "bolt":
            return ["target"]
        elif self.model_type == "chronos-2":
            return ["future_target"]
        else:
            return ["labels"]

# --- Training Execution ---

def train_and_evaluate_fold(fold_idx, train_data, val_data, config: TrainingConfig, train_cov=None, val_cov=None):
    print(f"\n=== Starting Fold {fold_idx+1}/{config.n_splits} ===")
    
    # 1. Initialize Model Manager and Pipeline
    manager = ChronosModelManager(config.model_name)
    pipeline = manager.load_pipeline()
    
    # 2. Prepare Datasets
    # Chronos-2 might need explicit prediction length if not in config
    pred_len = config.prediction_length
    
    train_dataset = DatasetFactory.get_dataset(
        manager.model_type, train_data, pipeline, config.context_length, pred_len, covariates=train_cov
    )
    val_dataset = DatasetFactory.get_dataset(
        manager.model_type, val_data, pipeline, config.context_length, pred_len, covariates=val_cov
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # 3. Setup LoRA
    lora_config = manager.get_lora_config()
    peft_model = manager.get_peft_model(lora_config)
    
    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=f"{config.output_dir}/{config.safe_model_name}/fold_{fold_idx}",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        # report_to="none", # Allow default reporting (e.g. Tensorboard)
        label_names=manager.get_label_names()
    )
    
    # 5. Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=chronos_collator
    )
    
    # 6. Train
    trainer.train()
    
    # 7. Save Adapter
    adapter_path = f"{config.output_dir}/{config.safe_model_name}/fold_{fold_idx}/final_adapter"
    # We can rely on the default saving, or explicitly save carefully
    peft_model.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path}")
    
    # 8. Evaluation
    print("Evaluating Fine-Tuned Model...")
    finetuned_results = evaluate_forecast(pipeline, val_data, config, val_cov=val_cov)
    
    # 9. Visualization & Comparison
    # We need the "Zero-Shot" (Original) results.
    # To do this cleanly, strict zero-shot should have been run BEFORE training.
    # However, since we are inside a function, we'd need to have done it earlier.
    # ALTERNATIVE: Unload/Reload pipeline? Or maintain two pipelines?
    # Simpler: We should have passed zero_shot_results to this function or run it before PEFT wrapping.
    
    # But wait, `pipeline` currently holds `peft_model`.
    # To get base model predictions, we can disable adapters temporarily.
    
    print("\nGenerating Comparison Plots...")
    with peft_model.disable_adapter():
        print("Evaluating Zero-Shot (Original) Model...")
        zeroshot_results = evaluate_forecast(pipeline, val_data, config, val_cov=val_cov)
        
    # Plot
    # We need history for MASE. That is `train_data`.
    plot_comparisons(
        train_data, 
        val_data, 
        zeroshot_results, 
        finetuned_results, 
        config, 
        fold_idx
    )
    
    # Print metrics summary
    print(f"\n=== Fold {fold_idx+1} Metrics Comparison ===")
    print(f"{'Metric':<10} | {'Original':<15} | {'Fine-Tuned':<15}")
    print("-" * 45)
    
    # Aggregated RMSE
    orig_rmse = np.mean([r['rmse'] for r in zeroshot_results])
    ft_rmse = np.mean([r['rmse'] for r in finetuned_results])
    print(f"{'RMSE':<10} | {orig_rmse:<15.4f} | {ft_rmse:<15.4f}")
    
    # Aggregated MAPE (if available)
    orig_mape = np.mean([r['mape'] for r in zeroshot_results if r['mape'] != np.inf])
    ft_mape = np.mean([r['mape'] for r in finetuned_results if r['mape'] != np.inf])
    print(f"{'MAPE':<10} | {orig_mape:<15.2f}% | {ft_mape:<15.2f}%")

def plot_comparisons(train_data, val_data, zero_shot_results, finetuned_results, config, fold_idx):
    """
    Plots the history, ground truth, and forecasts from both models.
    """
    item_ids = range(len(val_data)) # We assume list is aligned
    
    # Limit number of plots to avoid spam
    max_plots = 5
    indices_to_plot = item_ids[:max_plots]
    
    num_plots = len(indices_to_plot)
    if num_plots == 0: return

    fig, axs = plt.subplots(num_plots, 1, figsize=(15, 6 * num_plots), sharex=False)
    if num_plots == 1: axs = [axs]
    
    for i, idx in enumerate(indices_to_plot):
        ax = axs[i]
        
        # Ground Truth & History
        history = train_data[idx]
        val_seq = val_data[idx] # Contains context + target
        
        # Validation target starts at context_length
        target_start = config.context_length
        ground_truth = val_seq[target_start:]
        
        # Predictions
        zs_pred = zero_shot_results[idx]
        ft_pred = finetuned_results[idx]
        
        # Timesteps (Synthetic)
        hist_len = len(history)
        pred_len = config.prediction_length
        
        # We construct a synthetic timeline: History -> (Gap) -> Target
        # But wait, val_seq context IS the end of history?
        # Typically val_seq = history[-context:] + target
        # So history and val_seq overlap.
        
        # Plot full history
        ax.plot(range(hist_len), history, label='History', color='gray', alpha=0.5)
        
        # Plot Future Ground Truth
        # It starts at hist_len (assuming val starts right after train)
        future_time = range(hist_len, hist_len + pred_len)
        ax.plot(future_time, ground_truth, label='Ground Truth', color='black', linewidth=2)
        
        # Metrics
        # MASE needs history
        # We pass history from train_data
        
        mase_zs = calculate_mase(ground_truth, zs_pred['median'], history)
        mape_zs = calculate_mape(ground_truth, zs_pred['median'])
        
        mase_ft = calculate_mase(ground_truth, ft_pred['median'], history)
        mape_ft = calculate_mape(ground_truth, ft_pred['median'])
        
        # Plot Zero-Shot
        label_zs = f"Original | MASE: {mase_zs:.2f}, MAPE: {mape_zs:.1f}%"
        ax.plot(future_time, zs_pred['median'], label=label_zs, color='blue', linestyle='--')
        if 'quantiles' in zs_pred: # if we stored them
             pass # Fill between if available
             
        # Plot Fine-Tuned
        label_ft = f"Fine-Tuned | MASE: {mase_ft:.2f}, MAPE: {mape_ft:.1f}%"
        ax.plot(future_time, ft_pred['median'], label=label_ft, color='red', linestyle='--')
        
        ax.set_title(f"Fold {fold_idx+1} - Series {idx} Comparison", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    # Save the plot
    plot_path = os.path.join(config.output_dir, f"comparison_fold_{fold_idx}.png")
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")
    plt.close()

def evaluate_forecast(pipeline, val_data_list, config: TrainingConfig, val_cov=None) -> List[Dict]:
    """
    Returns a list of result dictionaries for each series:
    {
        'median': np.array,
        'rmse': float,
        'mape': float
    }
    """
    # print("Evaluating forecast...")
    results = []
    
    # Ensure input is list
    if not isinstance(val_data_list, list):
        val_data_list = [val_data_list]
    if val_cov is not None and not isinstance(val_cov, list):
        val_cov = [val_cov]
        
    # Prepare batch input for pipeline.predict if possible, or loop
    # Chronos pipeline predicts a list of inputs efficiently.
    
    inputs_for_pipeline = []
    ground_truths = []
    
    for idx, val_series in enumerate(val_data_list):
        # Need enough data for at least one prediction
        min_len = config.context_length + config.prediction_length
        if len(val_series) < min_len:
            continue

        val_series_torch = torch.tensor(val_series, dtype=torch.float32)
        
        # Take the last window for evaluation
        input_context = val_series_torch[-(config.context_length + config.prediction_length) : -config.prediction_length]
        ground_truth = val_series_torch[-config.prediction_length:].numpy()
        ground_truths.append(ground_truth)
        
        item_dict = {
            "target": input_context,
        }
        
        if val_cov is not None:
             # Grab corresponding covariate window
             current_cov = val_cov[idx]
             if len(current_cov) >= len(val_series):
                 # Align with the sliced context/truth
                 start_idx = len(val_series) - (config.context_length + config.prediction_length)
                 end_idx = len(val_series)
                 
                 full_cov_window = current_cov[start_idx : end_idx] # Shape: (TotalLen, Feats)
                 
                 # Split
                 past_cov_window = full_cov_window[:config.context_length]
                 future_cov_window = full_cov_window[config.context_length:]
                 
                 # Create Dicts
                 past_cov_dict = {}
                 future_cov_dict = {}
                 
                 for i, col_name in enumerate(config.covariate_cols):
                     # Extract feature i
                     past_cov_dict[col_name] = torch.tensor(past_cov_window[:, i], dtype=torch.float32)
                     future_cov_dict[col_name] = torch.tensor(future_cov_window[:, i], dtype=torch.float32)
                 
                 item_dict["past_covariates"] = past_cov_dict
                 item_dict["future_covariates"] = future_cov_dict
        
        inputs_for_pipeline.append(item_dict)

    if not inputs_for_pipeline:
        print("⚠️ No valid validation series found for evaluation.")
        return []

    # Bulk Predict
    # Note: Pipeline input is a list of dicts
    # print(f"Predicting for {len(inputs_for_pipeline)} series...")
    try:
        # Chronos2 pipeline expects specific dict structure
        forecasts = pipeline.predict(
            inputs_for_pipeline, 
            prediction_length=config.prediction_length,
        )
        # forecasts is a list of tensors (n_variates, n_quantiles, pred_len)
        
        for i, forecast in enumerate(forecasts):
            # Convert to Median
            # forecast shape: (1, quantiles, pred_len) for univariate
            # We assume median is the middle quantile or use a specific index if known.
            # Usually Chronos outputs quantiles=[0.1, ..., 0.5, ..., 0.9]
            # pipeline.quantiles property could tell us, but median is likely middle.
            
            # Simple median heuristic:
            forecast_np = forecast.numpy() # (samples/variates, quantiles, time)
            
            # Reduce variates dim (univariate)
            if forecast_np.shape[0] == 1:
                forecast_np = forecast_np[0]
            
            # Median along quantile dim
            # If we don't know which is median, we take valid median of values
            forecast_median = np.median(forecast_np, axis=0)
            
            # Calculate Metrics
            rmse = np.sqrt(mean_squared_error(ground_truths[i], forecast_median))
            mape = calculate_mape(ground_truths[i], forecast_median)
            
            results.append({
                "median": forecast_median,
                "rmse": rmse,
                "mape": mape
            })
            
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
    
    return results


# --- Main ---

def main():
    if not os.path.exists(CONFIG.output_dir):
        os.makedirs(CONFIG.output_dir)

    print(f"Loading data from {CONFIG.data_path}...")
    try:
        df = pd.read_parquet(CONFIG.data_path)
        
        # --- Multi-Series Handling ---
        # 1. Group by series_id
        if 'series_id' not in df.columns:
             print("Warning: 'series_id' column not found. Treating as single time series.")
             full_series_list = [df['pv'].values]
             full_cov_list = None
        else:
             print(f"Found {df['series_id'].nunique()} unique series.")
             groups = sorted(df.groupby('series_id'))
             full_series_list = [group['pv'].values for _, group in groups]
             
             # --- Load Covariates ---
             print(f"Looking for covariates: {CONFIG.covariate_cols}")
             present_cols = [c for c in CONFIG.covariate_cols if c in df.columns]
             missing_cols = [c for c in CONFIG.covariate_cols if c not in df.columns]
             
             if missing_cols:
                 print(f"⚠️ Warning: Missing covariate columns: {missing_cols}. They will be skipped.")
             
             if present_cols:
                 print(f"Loading covariates: {present_cols}")
                 # Create list of (Time, Features) arrays
                 full_cov_list = [group[present_cols].values for _, group in groups]
             else:
                 print("No requested covariates found.")
                 full_cov_list = None

        # 2. Split logic (Train/Val split per series)
        for fold_idx in range(CONFIG.n_splits):
             print(f"\nPreparing data for Fold {fold_idx + 1}...")
             
             train_series_list = []
             val_series_list = []
             train_cov_list = [] if full_cov_list else None
             val_cov_list = [] if full_cov_list else None
             
             for i, series in enumerate(full_series_list):
                 n_samples = len(series)
                 min_len = CONFIG.context_length + CONFIG.prediction_length
                 if n_samples < min_len:
                      continue

                 # Same split logic as before
                 validation_horizon = CONFIG.prediction_length
                 total_folds = CONFIG.n_splits
                 steps_back = (total_folds - fold_idx) * validation_horizon
                 train_end = n_samples - steps_back
                 val_end = train_end + validation_horizon
                 
                 if train_end < CONFIG.context_length:
                     continue
                     
                 train_seq = series[:train_end]
                 val_start_with_context = train_end - CONFIG.context_length
                 val_seq = series[val_start_with_context : val_end]
                 
                 train_series_list.append(train_seq)
                 val_series_list.append(val_seq)
                 
                 # Handle Covariates Split
                 if full_cov_list is not None:
                     cov_series = full_cov_list[i]
                     train_cov = cov_series[:train_end]
                     val_cov = cov_series[val_start_with_context : val_end]
                     
                     train_cov_list.append(train_cov)
                     val_cov_list.append(val_cov)
             
             if not train_series_list:
                 print(f"Skipping Fold {fold_idx}: No series long enough.")
                 continue
                 
             # Train
             train_and_evaluate_fold(
                 fold_idx, 
                 train_series_list, 
                 val_series_list, 
                 CONFIG, 
                 train_cov=train_cov_list, 
                 val_cov=val_cov_list
             )
            
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
