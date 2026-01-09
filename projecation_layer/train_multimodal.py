import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from peft import LoraConfig, get_peft_model, TaskType
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
    
    # Explicitly cast Vision Backbone to bfloat16 to save memory (it's frozen anyway)
    # print("Casting Vision Backbone to bfloat16 to save memory...")
    # if model.vision_backbone is not None:
    #     model.vision_backbone.to(dtype=torch.bfloat16)

    # Also cast the Projector to bfloat16 to match the incoming embeddings
    print("Casting Projector to bfloat16...")
    model.projector.to(dtype=torch.bfloat16)
    
    # Move entire fusion model to device
    model.to(device)
    
    # Apply LoRA to Chronos
    peft_config = LoraConfig(
        r=16,                    # Rank (same as default or adjust as needed)
        lora_alpha=32,           # Alpha (scaling factor)
        target_modules=[
            "self_attention.q",
            "self_attention.v",
            "self_attention.k",
            "self_attention.o",
            "output_patch_embedding.output_layer",
        ],
        lora_dropout=0.05,
        bias="none",
        #task_type="CAUSAL_LM",
        use_dora=True           
    )
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
