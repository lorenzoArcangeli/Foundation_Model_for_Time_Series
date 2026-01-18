import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from peft import LoraConfig, get_peft_model, TaskType, IA3Config, AdaLoraConfig, FourierFTConfig
import pandas as pd
import os
import io
import math
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Import our custom modules
from multimodal_chronos import MultimodalChronos
from multimodal_dataset import RandomMultimodalDataset, MultimodalDataset, collate_fn

# --- Metric & Plotting Helpers (Copied from run_visualization.py) ---
def calculate_item_mase(y_true, y_pred, y_history, seasonality=96):
    """Calculates MASE for a single item."""
    forecast_mae = np.mean(np.abs(y_true - y_pred))
    if len(y_history) <= seasonality:
        return np.inf 
    naive_errors = np.abs(y_history[seasonality:] - y_history[:-seasonality])
    naive_mae = np.mean(naive_errors)
    if naive_mae == 0:
        return np.inf
    return forecast_mae / naive_mae

def calculate_item_mape(y_true, y_pred, epsilon=1e-10):
    mask = y_true > epsilon
    if np.sum(mask) == 0:
        return np.nan 
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

def calculate_item_wmape(y_true, y_pred):
    total_abs_error = np.sum(np.abs(y_true - y_pred))
    total_actuals = np.sum(np.abs(y_true))
    if total_actuals == 0:
        return np.inf
    return (total_abs_error / total_actuals) * 100

def calculate_item_wql(y_true, quantile_preds, quantiles):
    total_loss = 0
    total_abs_target = np.sum(np.abs(y_true))
    if total_abs_target == 0:
        return np.inf
    for q in quantiles:
        y_pred_q = quantile_preds[q]
        errors = y_true - y_pred_q
        loss = np.maximum(q * errors, (q - 1) * errors)
        total_loss += np.sum(2 * loss)
    wql = total_loss / (len(quantiles) * total_abs_target)
    return wql

def calculate_item_sql(y_true, quantile_preds, quantiles, y_history, seasonality=96):
    total_loss = 0
    if len(y_history) <= seasonality:
        return np.inf
    naive_errors = np.abs(y_history[seasonality:] - y_history[:-seasonality])
    naive_mae = np.mean(naive_errors)
    if naive_mae == 0:
        return np.inf
    for q in quantiles:
        y_pred_q = quantile_preds[q]
        errors = y_true - y_pred_q
        loss = np.maximum(q * errors, (q - 1) * errors)
        total_loss += np.mean(2 * loss)
    sql = total_loss / (len(quantiles) * naive_mae)
    return sql

def log_validation_metrics(train_df, test_df, model_predictions, item_id, phase_info, output_dir, 
                          prediction_length=96, seasonality=96):
    
    # Setup Data
    full_data = pd.concat([train_df, test_df]).sort_values(['item_id', 'timestamp'])
    
    # Filter data for this specific item
    item_full_data = full_data[full_data['item_id'] == item_id].set_index('timestamp')
    item_test_data = test_df[test_df['item_id'] == item_id].set_index('timestamp')
    item_train_data = train_df[train_df['item_id'] == item_id].set_index('timestamp')

    history_context = item_full_data.iloc[-(200 + prediction_length):] # just for consistency if needed, but we don't plot
    ground_truth_future = item_test_data['pv_value']
    history_for_metric = item_train_data['pv_value'].values

    # Plot Each Model (Dict) -> We assume only one model "Multimodal" usually
    # If multiple models, we might want to log all? User example shows "Item X: ...", implied one model or aggregated?
    # User example: "- Item 0: MASE: ... "
    # If we have multiple models, we should probably output lines for each, or just the main one.
    # The current code passes `models_to_plot = {"Multimodal": pred_df}`.
    
    log_line = ""
    
    for idx, (model_name, pred_df_all) in enumerate(model_predictions.items()):
        item_preds = pred_df_all[pred_df_all['item_id'] == item_id].set_index('timestamp')
        
        if item_preds.empty:
            continue

        # Metrics
        try:
            y_pred_median = item_preds['predictions'].values[-len(ground_truth_future):]
            y_true = ground_truth_future.values[-len(y_pred_median):]

            mase_score = calculate_item_mase(y_true, y_pred_median, history_for_metric, seasonality)
            mape_score = calculate_item_mape(y_true, y_pred_median)
            wmape = calculate_item_wmape(y_true, y_pred_median)
            
            # WQL & SQL
            quantiles_to_check = [0.1, 0.5, 0.9]
            quantile_preds_dict = {}
            has_quantiles = True
            
            for q in quantiles_to_check:
                q_str = str(q)
                if q_str in item_preds.columns:
                    quantile_preds_dict[q] = item_preds[q_str].values[-len(ground_truth_future):]
                else:
                    has_quantiles = False
                    if q == 0.5: quantile_preds_dict[0.5] = y_pred_median
            
            if has_quantiles:
                wql_score = calculate_item_wql(y_true, quantile_preds_dict, quantiles_to_check)
                sql_score = calculate_item_sql(y_true, quantile_preds_dict, quantiles_to_check, history_for_metric, seasonality)
                metrics_str = (f"MASE: {mase_score:.2f} MAPE: {mape_score:.0f}% wMAPE: {wmape:.1f}% "
                                 f"WQL: {wql_score:.3f} SQL: {sql_score:.3f}")
            else:
                metrics_str = f"MASE: {mase_score:.2f} MAPE: {mape_score:.0f}% wMAPE: {wmape:.1f}%"

        except Exception as e:
            metrics_str = f"Error: {e}"

        item_log = f"- Item {item_id}: {metrics_str}"
        print(item_log) # Print to console
        log_line += item_log + "\n"

    return log_line

def run_validation_visualization(pipeline, model, df, step_name, output_dir, device="cuda"):
    """
    Calculates metrics and logs them to a text file.
    step_name example: "p1_10" -> Phase 1, Step 10
    """
    # ENSURE EVAL MODE (Disable Dropout/Noise)
    was_training = model.training
    model.eval()
    
    # Parse step_name for prettier logging
    phase_str = "UNKNOWN"
    step_num = "?"
    if step_name.startswith("p1_"):
        phase_str = "PHASE 1"
        step_num = step_name.split("_")[1]
    elif step_name.startswith("p2_"):
        phase_str = "PHASE 2"
        step_num = step_name.split("_")[1]
    
    header = f"{phase_str}:\nStep {step_num}:"

    # 1. Select items (e.g., first 3)
    distinct_items = sorted(df['item_id'].unique())
    selected_items = distinct_items[:3] # Visualize first 3 items
    
    # Filter DF for speed
    df_vis = df[df['item_id'].isin(selected_items)].copy()
    
    # 2. Project Features
    if 'visual_embedding' not in df_vis.columns:
        print("[Vis] Warning: 'visual_embedding' column not found. Skipping visualization.")
        return

    all_embeddings = np.stack(df_vis['visual_embedding'].values)
    all_tensor = torch.tensor(all_embeddings, dtype=torch.bfloat16, device=device)
    
    # Project
    with torch.no_grad():
        projected_tensor = model.projector(all_tensor)
        projected_np = projected_tensor.float().cpu().numpy()
        
    cov_cols = [f"cov_{i}" for i in range(model.covariate_dim)]
    cov_df = pd.DataFrame(projected_np, columns=cov_cols, index=df_vis.index)
    df_enriched = pd.concat([df_vis, cov_df], axis=1)

    if 'visual_embedding' in df_enriched.columns:
        df_enriched = df_enriched.drop(columns=['visual_embedding'])
        
    # 3. Predict using Pipeline
    
    # Split
    def split_vis(df_in):
        prediction_length = 96 # Hardcoded or global
        test_df = df_in.groupby('item_id').tail(prediction_length).copy()
        train_df = df_in.drop(test_df.index).copy()
        inference_df = test_df.copy()
        if 'pv_value' in inference_df.columns:
            inference_df = inference_df.drop(columns=['pv_value'])
        return train_df, inference_df, test_df

    train_df, inference_df, test_df = split_vis(df_enriched)
    
    # Predict
    pred_df = pipeline.predict_df(
        df=train_df,
        future_df=inference_df,
        context_length=CONTEXT_LENGTH, 
        prediction_length=96,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="item_id",
        timestamp_column="timestamp",
        target="pv_value",
    )
    
    # Log to file
    models_to_plot = {"Multimodal": pred_df}
    log_file_path = os.path.join(output_dir, "validation_metrics.txt")
    
    with open(log_file_path, "a") as f:
        f.write(f"\n{header}\n")
        for item_id in selected_items:
            line = log_validation_metrics(
                train_df=train_df,
                test_df=test_df,
                model_predictions=models_to_plot,
                item_id=item_id,
                phase_info=step_name,
                output_dir=output_dir
            )
            f.write(line)
            
    print(f"Logged validation metrics to {log_file_path}")
    
    # Restore State
    model.train(was_training)


# --- Configuration ---
DATA_PATH = r"c:\Users\loren\Desktop\D vecchio\UNIVERSITA\MAGISTRALE\SecondYear\FoundationModel\FM_test\project_features\datasets\skippd_train_embeddings.parquet"
OUTPUT_DIR = r"c:\Users\loren\Desktop\D vecchio\UNIVERSITA\MAGISTRALE\SecondYear\FoundationModel\FM_test\project_features\multimodal_infinite_checkpoints"
VISION_MODEL = "facebook/dinov2-small"
CHRONOS_MODEL = "amazon/chronos-2"

# Speed & Optimization
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4

# Step-based Training Config
MAX_STEPS = 50         # Total training steps (Phase 2)
WARMUP_STEPS_RATIO = 0.1 # 10% for OneCycleLR
PROJECTOR_WARMUP_STEPS = 100 # Phase 1: Train ONLY projector for this many steps
VAL_CHECK_INTERVAL = 10 # Validate every N steps
SAVE_INTERVAL = 10     # Save checkpoint every N steps

# Model Params
CONTEXT_LENGTH = 2048
PREDICTION_LENGTH = 96
COVARIATE_DIM = 20
MAX_GRAD_NORM = 1.0
PEFT_TYPE = "lora"

# Regularization
DROPOUT = 0.2 #0.1
NOISE_STD = 0.05 #0.01

def load_data(path):
    print(f"Loading dataset from {path}...")
    df = pd.read_parquet(path)
    
    if 'time' in df.columns and 'timestamp' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['time']) 
        if df['timestamp'].dt.tz is not None:
             df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    if "series_id" in df.columns:
        df = df.rename(columns={"series_id": "item_id", "pv": "pv_value"})

    # Drop 'time' if it exists to avoid type errors in Chronos pipeline (which dislikes TZ-aware columns)
    if 'time' in df.columns:
        df = df.drop(columns=['time'])

    df = df.sort_values(['item_id', 'timestamp']).reset_index(drop=True)
    return df

def get_peft_config(peft_type: str, target_modules: list, r: int = 16, lora_alpha: int = 32):
    """
    Factory function to create different PEFT configs.
    """
    peft_type = peft_type.lower()
    
    if peft_type == "lora":
        return LoraConfig(
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
            use_dora=True # Defaulting to DoRA as per original script's hardcoded config, but user can change if they want strictly LoRA
        )
    elif peft_type == "dora":
         return LoraConfig(
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
            use_dora=True 
        )
    elif peft_type == "ia3":
        # IA3 needs feedforward modules to be a subset of target_modules
        ff_modules = ["mlp.wi", "mlp.wo"]
        # Ensure target_modules includes feedforward if not already present
        combined_targets = list(set(target_modules + ff_modules))
        
        return IA3Config(
            inference_mode=False,
            target_modules=combined_targets, 
            feedforward_modules=ff_modules
        )
    elif peft_type == "adalora":
        return AdaLoraConfig(
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
        )
    elif peft_type == "fourierft":
        return FourierFTConfig(
            target_modules=target_modules,
            n_frequency=1000, # Default spectral count, can be tuned
            # scale=0.1 # Removed as it causes TypeError in current PEFT version
        )
    else:
        raise ValueError(f"Unsupported PEFT type: {peft_type}")

def validate(model, val_loader, device):
    """
    Runs evaluation loop on validation set (Deterministic).
    """
    model.eval()
    total_val_loss = 0
    steps = 0
    
    print("\n[Validation] Running Validation Loop...")
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for batch in val_loader:
                context = batch["context"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                future_target = None
                if batch["future_target"] is not None:
                     future_target = batch["future_target"].to(device)
                
                tabular_covariates = None
                if batch["tabular_covariates"] is not None:
                    tabular_covariates = batch["tabular_covariates"].to(device)
                
                outputs = model(
                    context_tensor=context,
                    pixel_values=pixel_values,
                    tabular_covariates=tabular_covariates,
                    future_target=future_target
                )
                
                loss = outputs.loss
                total_val_loss += loss.item()
                steps += 1
            
    avg_val_loss = total_val_loss / steps if steps > 0 else 0
    model.train() # Switch back to train mode
    return avg_val_loss

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Prepare Data
    df = load_data(DATA_PATH)
    
    # Use Full Dataset (No Split, reflecting `train_multimodal_steps.py` logic)
    print("Using full dataset for Training (and Validation as sanity check)...")
    df_train = df.copy()
    df_val = df.copy() # Validate on training data for sanity check since dataset is small
    
    print(f"Train Series: {df_train['item_id'].nunique()} | Val Series: {df_val['item_id'].nunique()} (Same Data)")
    
    # A. Infinite Random Train Dataset
    # We reserve the last PREDICTION_LENGTH steps of each series for validation
    train_dataset = RandomMultimodalDataset(
        df=df_train,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        image_processor=None,
        use_precomputed=True,
        reserved_end_steps=PREDICTION_LENGTH 
    )
    # Note: No shuffle=True needed for IterableDataset
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0) 
    
    # B. Deterministic Validation Dataset (Hold-out Future)
    val_dataset = MultimodalDataset(
        df=df_val,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        mode="validation", # Use validation mode to pick ONLY the last window
        use_precomputed=True,
        stride=1 # Ignored in validation mode
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 2. Prepare Model
    print("Initializing Model...")
    model = MultimodalChronos(
        chronos_model_name=CHRONOS_MODEL,
        vision_model_name=VISION_MODEL,
        covariate_dim=COVARIATE_DIM, 
        freeze_vision=False,
        use_precomputed_embeddings=True,
        dropout=DROPOUT,
        noise_std=NOISE_STD
    )
    
    # Check for PCA Init
    pca_init_path = os.path.join(OUTPUT_DIR, "vision_projector_pca_init.pth")
    if os.path.exists(pca_init_path):
        print(f"Loading PCA-Initialized Projector from {pca_init_path}...")
        model.projector.load_state_dict(torch.load(pca_init_path))
    else:
        print("No PCA initialization found. Using Random Init.")

    # Load Chronos
    from chronos import BaseChronosPipeline
    print("Loading Chronos Pipeline...")
    pipeline = BaseChronosPipeline.from_pretrained(CHRONOS_MODEL, device_map=device, torch_dtype=torch.bfloat16)
    model.chronos = pipeline.model
    model.projector.to(dtype=torch.bfloat16)
    model.to(device)
    
    # Setup LoRA / PEFT
    print(f"Applying PEFT Type: {PEFT_TYPE}")
    target_modules=[
        "self_attention.q",
        "self_attention.v",
        "self_attention.k",
        "self_attention.o",
        "output_patch_embedding.output_layer",
    ]
    
    peft_config = get_peft_config(PEFT_TYPE, target_modules, r=16, lora_alpha=32)
    model.chronos = get_peft_model(model.chronos, peft_config)
    pipeline.model = model.chronos # CRITICAL: Update pipeline to use the Peft model for prediction
    
    # Enable Gradient Checkpointing (Critical for VRAM)
    #print("Enabling Gradient Checkpointing...")
    #model.chronos.gradient_checkpointing_enable()
    # Required for gradient checkpointing with LoRA
    #model.chronos.enable_input_require_grads()
    
    # DEBUG: Verify LoRA targets
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    has_group_attn = any("group_self_attention" in n or ("layer.1" in n and "self_attention" in n) for n in trainable_names)
    print(f"DEBUG: LoRA captures GroupSelfAttention? {'YES' if has_group_attn else 'MAYBE (Check names manually)'}")
    # Note: 'layer.1' is usually GroupSelfAttention in Chronos2EncoderBlock
    
    # --- PHASE 1: PROJECTOR WARMUP ---
    print("\n=== PHASE 1: PROJECTOR WARMUP (Training ONLY Projector) ===")
    
    # Freeze Everything Except Projector
    for param in model.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = True
        
    optimizer_p1 = torch.optim.AdamW(model.projector.parameters(), lr=LEARNING_RATE)
    
    model.train()
    p1_steps = 0
    p1_iterator = iter(train_loader)
    
    # Phase 1 Accumulators
    p1_loss_sum = 0.0
    p1_count = 0
    
    while p1_steps < PROJECTOR_WARMUP_STEPS:
        try:
            batch = next(p1_iterator)
        except StopIteration:
            p1_iterator = iter(train_loader)
            batch = next(p1_iterator)
            
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            context = batch["context"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            future_target = batch["future_target"].to(device)
            
            tabular_covariates = None
            if batch["tabular_covariates"] is not None:
                tabular_covariates = batch["tabular_covariates"].to(device)

            outputs = model(
                context_tensor=context,
                pixel_values=pixel_values,
                tabular_covariates=tabular_covariates,
                future_target=future_target
            )
            loss = outputs.loss
        
        loss.backward()
        p1_loss_sum += loss.item()
        p1_count += 1
        
        if (p1_steps + 1) % 10 == 0:
             avg_p1_loss = p1_loss_sum / p1_count
             print(f"[Phase 1] Step {p1_steps+1}/{PROJECTOR_WARMUP_STEPS} | Avg Loss: {avg_p1_loss:.4f}")
             p1_loss_sum = 0.0
             p1_count = 0
             
        # Validation in Phase 1
        if (p1_steps + 1) % VAL_CHECK_INTERVAL == 0:
            val_loss = validate(model, val_loader, device)
            print(f"--> [Phase 1] Validation Step {p1_steps+1}: Loss {val_loss:.4f}")
            run_validation_visualization(pipeline, model, df, f"p1_{p1_steps+1}", OUTPUT_DIR, device=device)
             
        optimizer_p1.step()
        optimizer_p1.zero_grad()
        p1_steps += 1

    print("Phase 1 Complete. Projector Warmed Up.")
    
    # --- PHASE 2: JOINT TRAINING ---
    print("\n=== PHASE 2: JOINT TRAINING (LoRA + Projector) ===")
    
    # Unfreeze LoRA
    for n, p in model.chronos.named_parameters():
        if "lora" in n.lower():
            p.requires_grad = True
    
    # Verify Trainable Params
    model.chronos.print_trainable_parameters()
    
    trainable_params_p2 = [p for p in model.projector.parameters()] + \
                          [p for p in model.chronos.parameters() if p.requires_grad]
                          
    optimizer = torch.optim.AdamW(trainable_params_p2, lr=LEARNING_RATE)
    
    # OneCycleLR Logic for Steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE,
        total_steps=MAX_STEPS,
        pct_start=WARMUP_STEPS_RATIO
    )
    
    global_step = 0
    optimization_steps = 0

    total_loss = 0
    start_time = time.time()
    
    # Logging Accumulators
    interval_loss_sum = 0.0
    interval_step_count = 0
    
    # Refresh Iterator
    train_iterator = iter(train_loader)
    
    optimizer.zero_grad()
    
    while optimization_steps < MAX_STEPS:

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
            
        # Forward with Autocast
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            context = batch["context"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            future_target = batch["future_target"].to(device)
            
            tabular = None
            if batch["tabular_covariates"] is not None:
                tabular = batch["tabular_covariates"].to(device)
                
            outputs = model(
                context_tensor=context,
                pixel_values=pixel_values,
                tabular_covariates=tabular,
                future_target=future_target
            )
            loss = outputs.loss

        # Backward
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        current_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
        total_loss += current_loss
        interval_loss_sum += current_loss
        interval_step_count += 1
        
        # Optimizer Step
        if (global_step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params_p2, MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            optimization_steps += 1
            
            # Logging
            if optimization_steps % 10 == 0:
                 avg_loss = interval_loss_sum / interval_step_count if interval_step_count > 0 else 0.0
                 elapsed = time.time() - start_time
                 print(f"[Phase 2] Step {optimization_steps} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.1f}s")
                 
                 # Reset accumulators
                 interval_loss_sum = 0.0
                 interval_step_count = 0
                 start_time = time.time()
                 
            # Validation
            if optimization_steps % VAL_CHECK_INTERVAL == 0:
                val_loss = validate(model, val_loader, device)
                print(f"--> Validation Step {optimization_steps}: Loss {val_loss:.4f}")
                run_validation_visualization(pipeline, model, df, f"p2_{optimization_steps}", OUTPUT_DIR, device=device)
                
            # Checkpointing
            if optimization_steps % SAVE_INTERVAL == 0:
                save_path = os.path.join(OUTPUT_DIR, f"checkpoint_step_{optimization_steps}")
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.projector.state_dict(), os.path.join(save_path, "vision_projector.pth"))
                model.chronos.save_pretrained(os.path.join(save_path, "chronos_lora_adapter"))
                print(f"Saved Checkpoint to {save_path}")

        global_step += 1

    print("Training Complete!")
    
    # --- Final Save ---
    final_save_path = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_save_path, exist_ok=True)
    torch.save(model.projector.state_dict(), os.path.join(final_save_path, "vision_projector.pth"))
    model.chronos.save_pretrained(os.path.join(final_save_path, "chronos_lora_adapter"))
    print(f"Final Model Saved to {final_save_path}")

if __name__ == "__main__":
    main()
