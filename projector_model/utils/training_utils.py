import torch
import pandas as pd
import os
import numpy as np
from peft import LoraConfig, IA3Config, AdaLoraConfig, FourierFTConfig
from . import visualization_utils 
from . import config

def run_validation_visualization(pipeline, model, df, step_name, output_dir, context_length=config.CONTEXT_LENGTH, device="cuda"):
    """
    Calculates metrics and logs them to a text file.
    Wraps the visualization logic to be called from the training loop.
    """
    # Eval mode 
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

    # Select items (first 3)
    distinct_items = sorted(df['item_id'].unique())
    selected_items = distinct_items[:3] 
    
    # Filter DF for speed
    df_vis = df[df['item_id'].isin(selected_items)].copy()
    
    # Project Features
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
        
    # Predict using Pipeline
    
    # Simple split for visualization logic
    def split_vis(df_in):
        prediction_length = config.PREDICTION_LENGTH
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
        context_length=context_length, 
        prediction_length=config.PREDICTION_LENGTH,
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
            line = visualization_utils.log_validation_metrics(
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
    model.train() 
    return avg_val_loss

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
            use_dora=True
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
        ff_modules = ["mlp.wi", "mlp.wo"]
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
            n_frequency=1000, 
        )
    else:
        raise ValueError(f"Unsupported PEFT type: {peft_type}")
