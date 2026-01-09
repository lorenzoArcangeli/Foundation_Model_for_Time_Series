import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from peft import LoraConfig, get_peft_model, TaskType, IA3Config, AdaLoraConfig, FourierFTConfig
import pandas as pd
import os
import io
from PIL import Image

# Import our custom modules
from multimodal_chronos import MultimodalChronos
from multimodal_dataset import MultimodalDataset, collate_fn

# --- Configuration ---
DATA_PATH = r"c:\Users\loren\Desktop\D vecchio\UNIVERSITA\MAGISTRALE\SecondYear\FoundationModel\FM_test\project_features\skippd_train_embeddings.parquet"
OUTPUT_DIR = r"c:\Users\loren\Desktop\D vecchio\UNIVERSITA\MAGISTRALE\SecondYear\FoundationModel\FM_test\project_features\multimodal_checkpoints"
VISION_MODEL = "facebook/dinov2-small"
CHRONOS_MODEL = "amazon/chronos-2"
BATCH_SIZE = 16 # Reduced because Group Attention expands batch by 17x
GRADIENT_ACCUMULATION_STEPS = 2 # Accumulate to effective batch of 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
CONTEXT_LENGTH = 192 # Reduced to ~4 days for efficiency
PREDICTION_LENGTH = 96
COVARIATE_DIM = 16
STRIDE = 1 # 12 hours between samples (48 steps/day)
PEFT_TYPE = "dora" # "lora", "dora", "ia3", "adalora", "fourierft"

def load_data(path):
    print(f"Loading dataset from {path}...")
    df = pd.read_parquet(path)
    
    # Ensuring timestamp sorting etc.
    if 'time' in df.columns and 'timestamp' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['time']) 
        if df['timestamp'].dt.tz is not None:
             df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    # Rename for consistency with Dataset class expectations
    # Check if we need to rename (precomputed file might have 'series_id' and 'pv')
    if "series_id" in df.columns:
        df = df.rename(columns={"series_id": "item_id", "pv": "pv_value"})

    df = df.sort_values(['item_id', 'timestamp']).reset_index(drop=True)
    return df

def get_peft_config(peft_type: str, target_modules: list, r: int = 8, lora_alpha: int = 16):
    """
    Factory function to create different PEFT configs.
    """
    peft_type = peft_type.lower()
    
    if peft_type == "lora":
        return LoraConfig(
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
        )
    elif peft_type == "dora":
        return LoraConfig(
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
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
            scale=0.1         # Default scale
        )
    else:
        raise ValueError(f"Unsupported PEFT type: {peft_type}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Prepare Data
    df = load_data(DATA_PATH)
    
    # Dataset
    # image_processor is not needed if using precomputed embeddings
    dataset = MultimodalDataset(
        df=df,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        image_processor=None,
        mode="train",
        use_precomputed=True,
        stride=STRIDE
    )
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    print(f"Dataset created with {len(dataset)} samples.")

    # 2. Prepare Model
    model = MultimodalChronos(
        chronos_model_name=CHRONOS_MODEL,
        vision_model_name=VISION_MODEL,
        covariate_dim=COVARIATE_DIM, 
        freeze_vision=True,
        use_precomputed_embeddings=True
    )
    
    # Load PCA Initialization if available
    pca_init_path = os.path.join(OUTPUT_DIR, "vision_projector_pca_init.pth")
    if os.path.exists(pca_init_path):
        print(f"Loading PCA-Initialized Projector from {pca_init_path}...")
        model.projector.load_state_dict(torch.load(pca_init_path))
    else:
        print("No PCA initialization found. Training projector from scratch (Random Init).")
    
    # Load Chronos part correctly
    from chronos import BaseChronosPipeline
    print("Loading Chronos Pipeline to get model...")
    pipeline = BaseChronosPipeline.from_pretrained(CHRONOS_MODEL, device_map=device, torch_dtype=torch.bfloat16)
    model.chronos = pipeline.model
    
    # Also cast the Projector to bfloat16 to match the incoming embeddings
    print("Casting Projector to bfloat16...")
    model.projector.to(dtype=torch.bfloat16)
    
    # Move entire fusion model to device
    model.to(device)
    
    # Apply PEFT to Chronos
    print(f"Applying PEFT Type: {PEFT_TYPE}")
    target_modules=[
        "self_attention.q",
        "self_attention.v",
        "self_attention.k",
        "self_attention.o",
        "output_patch_embedding.output_layer",
    ]
    
    peft_config = get_peft_config(PEFT_TYPE, target_modules, r=16, lora_alpha=32)

    """
    # Apply LoRA to Chronos
    peft_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "self_attention.q",
            "self_attention.v",
            "self_attention.k",
            "self_attention.o",
            "output_patch_embedding.output_layer",
        ],
    )
    """
    # We wrap just the chronos component
    model.chronos = get_peft_model(model.chronos, peft_config)
    model.chronos.print_trainable_parameters()
    
    # 3. Optimizer
    trainable_params = [p for p in model.projector.parameters()] + \
                       [p for p in model.chronos.parameters() if p.requires_grad]
                       
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    
    # 4. Training Loop
    model.train()
    print("Starting Training...")
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        steps = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(dataloader):
            # Move data to device and cast to bfloat16
            context = batch["context"].to(device).bfloat16() 
            pixel_values = batch["pixel_values"].to(device).bfloat16() 
            future_target = batch["future_target"].to(device).bfloat16()
            
            tabular_covariates = None
            if batch["tabular_covariates"] is not None:
                tabular_covariates = batch["tabular_covariates"].to(device).bfloat16()
            
            # Forward
            outputs = model(
                context_tensor=context,
                pixel_values=pixel_values,
                tabular_covariates=tabular_covariates,
                future_target=future_target
            )
            
            # Loss calculation
            loss = outputs.loss if hasattr(outputs, "loss") else None
            
            if loss is None:
                print("Warning: Model didn't return loss. Check implementation.")
                break
            
            # Normalize loss for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Scale back for logging
            current_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
            total_loss += current_loss
            steps += 1
            
            # Aggressive cleanup
            del context, pixel_values, future_target, outputs, loss
            
            if steps % 10 == 0:
                print(f"Epoch {epoch+1} | Step {steps} | Loss: {current_loss:.4f}")
                # Optional: empty cache if really tight
                # torch.cuda.empty_cache() 
                
        print(f"Epoch {epoch+1} Complete. Avg Loss: {total_loss/steps:.4f}")
        
        # Save Checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Saving Checkpoint for Epoch {epoch+1}...")
            epoch_dir = os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            
            # Save Projector
            torch.save(model.projector.state_dict(), os.path.join(epoch_dir, "vision_projector.pth"))
            # Save Chronos Adapter
            model.chronos.save_pretrained(os.path.join(epoch_dir, "chronos_lora_adapter"))
            print(f"Checkpoint saved to {epoch_dir}")
        
    # 5. Save
    print("Saving Projector and Adapter...")
    torch.save(model.projector.state_dict(), os.path.join(OUTPUT_DIR, "vision_projector.pth"))
    model.chronos.save_pretrained(os.path.join(OUTPUT_DIR, "chronos_lora_adapter"))
    print("Done!")

if __name__ == "__main__":
    main()
