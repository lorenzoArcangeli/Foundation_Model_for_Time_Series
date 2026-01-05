import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import TimeSeriesSplit
from chronos import ChronosPipeline, BaseChronosPipeline
from torch.utils.data import default_collate
from chronos.chronos2.pipeline import Chronos2Pipeline
import math


def select_best_adapter_folder(model_name):
  safe_name = model_name.replace("/", "_").replace("-", "_")
  CHECKPOINT_BASE = f"/content/chronos_finetuning_checkpoints/{safe_name}"

  fold_dirs = [
      d for d in os.listdir(CHECKPOINT_BASE)
      if d.startswith("fold_") and os.path.isdir(os.path.join(CHECKPOINT_BASE, d))
  ]

  # 2. Sort them numerically by the fold number (handles fold_2 vs fold_10 correctly)
  # We split 'fold_3' by '_' to get ['fold', '3'], then take the last part and convert to int
  if fold_dirs:
      fold_dirs.sort(key=lambda x: int(x.split('_')[-1]))

      # 3. Pick the last one (highest number)
      latest_fold = fold_dirs[-1]

      adapter_path = os.path.join(CHECKPOINT_BASE, latest_fold, "final_adapter")

      print(f"Selected adapter: {adapter_path}")
  else:
      raise FileNotFoundError(f"No 'fold_*' directories found in {CHECKPOINT_BASE}")
  return adapter_path

  
# Configuration
DATA_PATH =  "/content/drive/MyDrive/FM_project/dataset/skippd_train_no_images.parquet"
# MODEL_NAME = "amazon/chronos-bolt-small" 
# MODEL_NAME = "amazon/chronos-t5-small"
MODEL_NAME = "amazon/chronos-2" # Chronos-2
CONTEXT_LENGTH = 600
PREDICTION_LENGTH = 60
N_SPLITS = 2
OUTPUT_DIR = "chronos_finetuning_checkpoints"


# Safer way to get patch size
def get_patch_size(model):
    if hasattr(model, "chronos_config"):
        return getattr(model.chronos_config, "output_patch_size", 16)
    elif hasattr(model, "config") and hasattr(model.config, "chronos_config"):
        return getattr(model.config.chronos_config, "output_patch_size", 16)
    return 16 # Fallback
    
class ChronosDataset(Dataset):
    def __init__(self, time_series, pipeline, context_length, prediction_length=None, model_type="t5"):
        self.time_series = time_series
        self.pipeline = pipeline
        self.context_length = context_length
        self.model_type = model_type # "t5", "bolt", "chronos-2"
        
        # Use model's expected prediction length if not provided
        if self.model_type == "chronos-2":
             # Chronos-2 config does not have prediction_length (it's flexible/patch-based)
             # So we rely on the user-provided prediction_length
             self.prediction_length = prediction_length or 64 # Default fallback
             
             # Need output_patch_size to calc num_output_patches
             self.output_patch_size = get_patch_size(pipeline.model)

        elif hasattr(pipeline.model, "chronos_config"):
             self.prediction_length = pipeline.model.chronos_config.prediction_length
        else:
             self.prediction_length = pipeline.model.config.prediction_length

        if prediction_length is not None and prediction_length != self.prediction_length:
            print(f"Warning: Requested prediction_length {prediction_length} differs from model's default {self.prediction_length}. Using model's default {self.prediction_length if self.model_type != 'chronos-2' else prediction_length}")
            if self.model_type == "chronos-2":
                 self.prediction_length = prediction_length
        
        # Tokenizer is only for T5
        self.tokenizer = getattr(pipeline, "tokenizer", None)
        
        # Create sliding windows
        self.samples = []
        total_window = self.context_length + self.prediction_length
        
        if isinstance(time_series, torch.Tensor):
            time_series = time_series.numpy()
            
        for i in range(len(time_series) - total_window + 1):
            window = time_series[i : i + total_window]
            self.samples.append(window)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window = self.samples[idx]
        #context = torch.tensor(window[:self.context_length])
        #target = torch.tensor(window[self.context_length:])
        
        if self.model_type in ["bolt", "chronos-2"]:
            # 1. Create Masks: True (1) for real data, False (0) for NaNs
            context_mask = ~torch.isnan(raw_context)
            target_mask = ~torch.isnan(raw_target)

            # 2. Fill NaNs: Replace NaNs with 0.0 so the model doesn't crash.
            #    The model will use the mask to ignore these 0.0s during attention/loss calc.
            context = torch.nan_to_num(raw_context, nan=0.0)
            target = torch.nan_to_num(raw_target, nan=0.0)
            
            if self.model_type == "chronos-2":
                 # Chronos-2 expects 'future_target' instead of 'target'
                 # And 'num_output_patches'
                 num_patches = math.ceil(self.prediction_length / self.output_patch_size)
                 
                 return {
                    "context": context,
                    "future_target": target, 
                    "context_mask": context_mask,
                    "future_target_mask": target_mask,
                    "num_output_patches": num_patches
                 }
            else:
                 # Bolt Specifics:
                 # - Requires 'target' key
                 # - Requires context mask named simply 'mask'
                 return {
                    "context": context, 
                    "target": target,
                    "mask": context_mask, 
                    "target_mask": target_mask
                 }
        else:
            # T5 (Original Chronos) behavior: Tokenization
            context_batch = context.unsqueeze(0) 
            target_batch = target.unsqueeze(0)   
            
            input_ids, attention_mask, tokenizer_state = self.tokenizer.context_input_transform(context_batch)
            labels, labels_mask = self.tokenizer.label_input_transform(target_batch, tokenizer_state)
            
            return {
                "input_ids": input_ids.squeeze(0),
                "attention_mask": attention_mask.squeeze(0),
                "labels": labels.squeeze(0)
            }


def chronos_collator(batch):
    # Separate num_output_patches if present
    num_output_patches = None
    clean_batch = []
    
    for item in batch:
        item_copy = item.copy()
        if "num_output_patches" in item_copy:
             val = item_copy.pop("num_output_patches")
             if num_output_patches is None:
                 num_output_patches = val
        clean_batch.append(item_copy)
    
    # Collate the rest using default
    collated = default_collate(clean_batch)
    
    # Re-insert num_output_patches as scalar int if found
    if num_output_patches is not None:
        collated["num_output_patches"] = int(num_output_patches)
        
    return collated

def train_and_evaluate_fold(fold_idx, train_data, val_data):
    # ...
    print(f"\n=== Starting Fold {fold_idx+1}/{N_SPLITS} ===")
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Check model type
    is_bolt = "bolt" in MODEL_NAME
    is_chronos2 = "chronos-2" in MODEL_NAME or "710m-2" in MODEL_NAME
    
    pipeline_class = ChronosPipeline # Default
    model_type = "t5"
    
    if is_bolt:
        pipeline_class = BaseChronosPipeline
        model_type = "bolt"
        print(f"Loading Bolt pipeline for {MODEL_NAME}")
    elif is_chronos2:
        if Chronos2Pipeline is None:
            raise ImportError("Chronos2Pipeline not found. Ensure chronos-forecasting is installed correctly.")
        pipeline_class = Chronos2Pipeline
        model_type = "chronos-2"
        print(f"Loading Chronos-2 pipeline for {MODEL_NAME}")
    else:
        print(f"Loading T5 pipeline for {MODEL_NAME}")

    pipeline = pipeline_class.from_pretrained(
        MODEL_NAME,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )
    
    # 2. Prepare Datasets
    train_dataset = ChronosDataset(train_data, pipeline, CONTEXT_LENGTH, PREDICTION_LENGTH, model_type=model_type)
    val_dataset = ChronosDataset(val_data, pipeline, CONTEXT_LENGTH, PREDICTION_LENGTH, model_type=model_type)
    
    # 3. LoRA Configuration
    # Bolt/Chronos-2 don't expose standard generation methods expected by PEFT's SEQ_2_SEQ_LM
    # So we use None or a generic type to avoid the AttributeError
    task_type = TaskType.SEQ_2_SEQ_LM if (not is_bolt and not is_chronos2) else None
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # 2.5% of the parameters
        target_modules=["q", "v", "k", "o"], #target_modules=["q", "v"] → 1.25% of the parameters
        lora_dropout=0.05,
        bias="none",
        task_type=task_type
    )
    
    # 4. Wrap Model with PEFT
    # T5: pipeline.model (ChronosModel) -> .model (HF T5)
    # Bolt: pipeline.model (ChronosBoltModel)
    # Chronos-2: pipeline.model (Chronos2Model)
    if is_bolt:
        model_to_train = pipeline.model
    elif is_chronos2:
        model_to_train = pipeline.model # Chronos2Model typically exposed as pipeline.model directly
    else:
        model_to_train = pipeline.model.model
        
    peft_model = get_peft_model(model_to_train, lora_config)
    peft_model.print_trainable_parameters()
    
    # 5. Training Arguments
    # Trainer needs to know the label key for evaluation metrics/loss logging
    if is_bolt:
        label_names = ["target"]
    elif is_chronos2:
        label_names = ["future_target"]
    else:
        label_names = ["labels"]
    

    safe_name = MODEL_NAME.replace("/", "_").replace("-", "_")
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/{safe_name}/fold_{fold_idx}",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=1, 
        learning_rate=1e-4,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=False, 
        bf16=torch.cuda.is_bf16_supported(),
        remove_unused_columns=False, 
        report_to="none",
        label_names=label_names # Crucial for Bolt/Chronos-2 evaluation
    )
    
    # 6. Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=chronos_collator 
    )
    
    trainer.train()
    
    # Save Model
    adapter_path = select_best_adapter_folder(MODEL_NAME)
    peft_model.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path}")


    from sklearn.metrics import mean_squared_error
    
    print("Evaluating forecast...")
    # 1. Get Forecast
    val_series_torch = torch.tensor(val_data, dtype=torch.float32)
    input_context = val_series_torch[-(CONTEXT_LENGTH + PREDICTION_LENGTH) : -PREDICTION_LENGTH]
    ground_truth = val_series_torch[-PREDICTION_LENGTH:].numpy()
    
    # Predict
    forecast = pipeline.predict(input_context.unsqueeze(0), prediction_length=PREDICTION_LENGTH)
    
    # 2. Convert to Median (Unified format for all models)
    # Chronos forecasts are usually shape (num_samples, prediction_length)
    forecast_median = np.median(forecast.numpy(), axis=1)[0]
    
    # 3. Calculate RMSE
    rmse = np.sqrt(mean_squared_error(ground_truth, forecast_median))
    
    print(f"✅ FINAL RESULTS for {MODEL_NAME}")
    print(f"📉 RMSE: {rmse:.4f}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    full_series = df['pv'].values

    limit = int(len(full_series) * 0.5)
    full_series = full_series[-limit:]
    print(f"Total time series length (truncated to 50%): {len(full_series)}")
    
    # Cross-Validation
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    for fold_idx, (train_index, val_index) in enumerate(tscv.split(full_series)):
        # Important: val_index in TimeSeriesSplit is usually small.
        # We need to ensure we have enough data for at least one context window + prediction
        
        train_data = full_series[train_index]
        val_data = full_series[val_index]
        
        # Augment validation data with enough context from end of train
        # The validation set naturally follows training, but to predict the *first* point of val
        # we need context from train.
        
        val_context_buffer = train_data[-CONTEXT_LENGTH:]
        val_data_extended = np.concatenate([val_context_buffer, val_data])
        
        train_and_evaluate_fold(fold_idx, train_data, val_data_extended)

if __name__ == "__main__":
    main()
