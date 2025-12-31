import os
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, default_collate
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
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
    
    @property
    def safe_model_name(self):
        return self.model_name.replace("/", "_").replace("-", "_")

# Global config instance (can be made dynamic via CLI args later if needed)
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

# --- Dataset Handler ---

class BaseChronosDataset(Dataset):
    def __init__(self, time_series, pipeline, context_length, prediction_length, model_type):
        self.time_series = time_series
        self.pipeline = pipeline
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.model_type = model_type
        
        # Prepare sliding windows
        self.samples = []
        total_window = self.context_length + self.prediction_length
        
        if isinstance(time_series, torch.Tensor):
            time_series = time_series.numpy()
            
        # Create samples
        for i in range(len(time_series) - total_window + 1):
            window = time_series[i : i + total_window]
            self.samples.append(window)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__")


class ChronosT5Dataset(BaseChronosDataset):
    def __init__(self, time_series, pipeline, context_length, prediction_length):
        super().__init__(time_series, pipeline, context_length, prediction_length, "t5")
        self.tokenizer = pipeline.tokenizer
        
        # Determine prediction_length from model config if not explicitly overridden/handled
        # For T5, we usually respect the model's config, but if user asked for specific prediction_length,
        # we might be limited by the model's max capacity. Here we assume training setup matches.

    def __getitem__(self, idx):
        window = self.samples[idx]
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
    def __init__(self, time_series, pipeline, context_length, prediction_length):
        super().__init__(time_series, pipeline, context_length, prediction_length, "bolt")

    def __getitem__(self, idx):
        window = self.samples[idx]
        context = torch.tensor(window[:self.context_length])
        target = torch.tensor(window[self.context_length:])
        
        # Bolt expects raw tensors and boolean masks
        return {
            "context": context,
            "target": target,
            "mask": torch.ones_like(context, dtype=torch.bool),
            "target_mask": torch.ones_like(target, dtype=torch.bool)
        }


class Chronos2Dataset(BaseChronosDataset):
    def __init__(self, time_series, pipeline, context_length, prediction_length):
        super().__init__(time_series, pipeline, context_length, prediction_length, "chronos-2")
        self.output_patch_size = get_patch_size(pipeline.model)

    def __getitem__(self, idx):
        window = self.samples[idx]
        context = torch.tensor(window[:self.context_length])
        target = torch.tensor(window[self.context_length:])
        
        # Chronos-2 uses 'future_target' and needs 'num_output_patches'
        num_patches = math.ceil(self.prediction_length / self.output_patch_size)
        
        return {
            "context": context,
            "future_target": target,
            "context_mask": torch.ones_like(context, dtype=torch.bool),
            "future_target_mask": torch.ones_like(target, dtype=torch.bool),
            "num_output_patches": num_patches
        }

class DatasetFactory:
    @staticmethod
    def get_dataset(model_type: str, *args, **kwargs) -> BaseChronosDataset:
        if model_type == "t5":
            return ChronosT5Dataset(*args, **kwargs)
        elif model_type == "bolt":
            return ChronosBoltDataset(*args, **kwargs)
        elif model_type == "chronos-2":
            return Chronos2Dataset(*args, **kwargs)
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

def train_and_evaluate_fold(fold_idx, train_data, val_data, config: TrainingConfig):
    print(f"\n=== Starting Fold {fold_idx+1}/{config.n_splits} ===")
    
    # 1. Initialize Model Manager and Pipeline
    manager = ChronosModelManager(config.model_name)
    pipeline = manager.load_pipeline()
    
    # 2. Prepare Datasets
    # Chronos-2 might need explicit prediction length if not in config
    pred_len = config.prediction_length
    
    train_dataset = DatasetFactory.get_dataset(
        manager.model_type, train_data, pipeline, config.context_length, pred_len
    )
    val_dataset = DatasetFactory.get_dataset(
        manager.model_type, val_data, pipeline, config.context_length, pred_len
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
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        report_to="none",
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
    # Use our helper function logic style to mimic original behavior or just direct save
    peft_model.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path}")
    
    # 8. Evaluation (Quick Forecast check)
    evaluate_forecast(pipeline, val_data, config)

def evaluate_forecast(pipeline, val_data, config: TrainingConfig):
    print("Evaluating forecast...")
    val_series_torch = torch.tensor(val_data, dtype=torch.float32)
    
    # Take the last window for evaluation
    input_context = val_series_torch[-(config.context_length + config.prediction_length) : -config.prediction_length]
    ground_truth = val_series_torch[-config.prediction_length:].numpy()
    
    # Predict
    forecast = pipeline.predict(input_context.unsqueeze(0), prediction_length=config.prediction_length)
    
    # Convert to Median
    forecast_median = np.median(forecast.numpy(), axis=1)[0]
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(ground_truth, forecast_median))
    print(f"✅ FINAL RESULTS for {config.model_name}")
    print(f"📉 RMSE: {rmse:.4f}")

# --- Main ---

def main():
    if not os.path.exists(CONFIG.output_dir):
        os.makedirs(CONFIG.output_dir)

    print(f"Loading data from {CONFIG.data_path}...")
    try:
        df = pd.read_parquet(CONFIG.data_path)
        full_series = df['pv'].values
        
        # Truncate for testing/speed if needed (preserving original logic)
        limit = int(len(full_series) * 0.5)
        full_series = full_series[-limit:]
        print(f"Total time series length (truncated to 50%): {len(full_series)}")
        
        tscv = TimeSeriesSplit(n_splits=CONFIG.n_splits)
        
        for fold_idx, (train_index, val_index) in enumerate(tscv.split(full_series)):
            train_data = full_series[train_index]
            val_data = full_series[val_index]
            
            # Augment validation data with context
            val_context_buffer = train_data[-CONFIG.context_length:]
            val_data_extended = np.concatenate([val_context_buffer, val_data])
            
            train_and_evaluate_fold(fold_idx, train_data, val_data_extended, CONFIG)
            
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
