import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from peft import get_peft_model
import os
import time

# Custom Imports
from multimodal_chronos import MultimodalChronos
from multimodal_dataset import RandomMultimodalDataset, MultimodalDataset, collate_fn
from utils import data_utils
from utils import training_utils

from utils import config

# --- Configuration ---
# Uses config.py for localized settings

def main():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Prepare Data
    df = data_utils.load_data(config.DATA_PATH)
    
    # Using full dataset for training, validating on hold-out windows of the same set
    print("Using full dataset for Training...")
    df_train = df.copy()
    df_val = df.copy() 
    
    print(f"Series: {df_train['item_id'].nunique()}")
    
    # A. Infinite Random Train Dataset
    train_dataset = RandomMultimodalDataset(
        df=df_train,
        prediction_length=config.PREDICTION_LENGTH,
        context_length=config.CONTEXT_LENGTH,
        image_processor=None,
        use_precomputed=True,
        reserved_end_steps=config.PREDICTION_LENGTH 
    )
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, num_workers=0) 
    
    # B. Deterministic Validation Dataset
    val_dataset = MultimodalDataset(
        df=df_val,
        prediction_length=config.PREDICTION_LENGTH,
        context_length=config.CONTEXT_LENGTH,
        mode="validation", 
        use_precomputed=True,
        stride=1 
    )
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 2. Prepare Model
    print("Initializing Model...")
    model = MultimodalChronos(
        chronos_model_name=config.CHRONOS_MODEL,
        vision_model_name=config.VISION_MODEL,
        covariate_dim=config.COVARIATE_DIM, 
        freeze_vision=False,
        use_precomputed_embeddings=True,
        dropout=0.2, # Hardcoded or add to config if desired
        noise_std=0.05, # Hardcoded or add to config
        device_map=device,
        torch_dtype=torch.bfloat16
    )
    
    # Check for PCA Init
    pca_init_path = os.path.join(config.CHECKPOINT_DIR, "vision_projector_pca_init.pth")
    if os.path.exists(pca_init_path):
        print(f"Loading PCA-Initialized Projector from {pca_init_path}...")
        model.projector.load_state_dict(torch.load(pca_init_path))
    else:
        print("No PCA initialization found. Using Random Init.")

    # Access pipeline from the model
    pipeline = model.pipeline
    model.projector.to(dtype=torch.bfloat16)
    # model.to(device) # already handled by device_map for chronos, ensure projector is on device
    model.projector.to(device)
    
    # Setup LoRA / PEFT
    print(f"Applying PEFT Type: {config.PEFT_TYPE}")
    target_modules=[
        "self_attention.q",
        "self_attention.v",
        "self_attention.k",
        "self_attention.o",
        "output_patch_embedding.output_layer",
    ]
    
    peft_config = training_utils.get_peft_config(config.PEFT_TYPE, target_modules, r=16, lora_alpha=32)
    model.chronos = get_peft_model(model.chronos, peft_config)
    pipeline.model = model.chronos 
    
    # DEBUG: Verify LoRA targets
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    has_group_attn = any("group_self_attention" in n or ("layer.1" in n and "self_attention" in n) for n in trainable_names)
    print(f"DEBUG: LoRA captures GroupSelfAttention? {'YES' if has_group_attn else 'MAYBE (Check names manually)'}")
    
    # --- PHASE 1: PROJECTOR WARMUP ---
    print("\n=== PHASE 1: PROJECTOR WARMUP (Training ONLY Projector) ===")
    
    # Freeze Everything Except Projector
    for param in model.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = True
        
    optimizer_p1 = torch.optim.AdamW(model.projector.parameters(), lr=config.LEARNING_RATE)
    
    model.train()
    p1_steps = 0
    p1_iterator = iter(train_loader)
    
    # Phase 1 Accumulators
    p1_loss_sum = 0.0
    p1_count = 0
    
    # Constants locally for loop
    PROJECTOR_WARMUP_STEPS = 150 # Or from config
    VAL_CHECK_INTERVAL = 10
    
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
            if True: # ENABLE_VISUALIZATION
                 training_utils.run_validation_visualization(pipeline, model, df, f"p1_{p1_steps+1}", config.CHECKPOINT_DIR, context_length=config.CONTEXT_LENGTH, device=device)
             
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
                          
    optimizer = torch.optim.AdamW(trainable_params_p2, lr=config.LEARNING_RATE)
    
    # OneCycleLR Logic for Steps
    MAX_STEPS = 100 # Or from config
    WARMUP_STEPS_RATIO = 0.1

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config.LEARNING_RATE,
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
    
    train_iterator = iter(train_loader)
    
    optimizer.zero_grad()
    
    while optimization_steps < MAX_STEPS:

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
            
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

        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        current_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
        total_loss += current_loss
        interval_loss_sum += current_loss
        interval_step_count += 1
        
        if (global_step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params_p2, config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            optimization_steps += 1
            
            if optimization_steps % 10 == 0:
                 avg_loss = interval_loss_sum / interval_step_count if interval_step_count > 0 else 0.0
                 elapsed = time.time() - start_time
                 print(f"[Phase 2] Step {optimization_steps} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.1f}s")
                 
                 interval_loss_sum = 0.0
                 interval_step_count = 0
                 start_time = time.time()
                 
            SAVE_INTERVAL = 10
            
            if optimization_steps % VAL_CHECK_INTERVAL == 0:
                val_loss = training_utils.validate(model, val_loader, device)
                print(f"--> Validation Step {optimization_steps}: Loss {val_loss:.4f}")
                if True: # ENABLE_VISUALIZATION
                    training_utils.run_validation_visualization(pipeline, model, df, f"p2_{optimization_steps}", config.CHECKPOINT_DIR, context_length=config.CONTEXT_LENGTH, device=device)
                
            if optimization_steps % SAVE_INTERVAL == 0:
                save_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_step_{optimization_steps}")
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.projector.state_dict(), os.path.join(save_path, "vision_projector.pth"))
                model.chronos.save_pretrained(os.path.join(save_path, "chronos_lora_adapter"))
                print(f"Saved Checkpoint to {save_path}")

        global_step += 1

    print("Training Complete!")
    
    final_save_path = os.path.join(config.CHECKPOINT_DIR, "final_model")
    os.makedirs(final_save_path, exist_ok=True)
    torch.save(model.projector.state_dict(), os.path.join(final_save_path, "vision_projector.pth"))
    model.chronos.save_pretrained(os.path.join(final_save_path, "chronos_lora_adapter"))
    print(f"Final Model Saved to {final_save_path}")

if __name__ == "__main__":
    main()
