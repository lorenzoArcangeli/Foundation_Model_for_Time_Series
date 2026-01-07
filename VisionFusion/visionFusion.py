import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForImageClassification, AutoImageProcessor
from chronos import BaseChronosPipeline
from peft import LoraConfig, get_peft_model, PeftModel
import pandas as pd
import numpy as np
from PIL import Image
import io
import os
from tqdm import tqdm

# --- Configuration ---
BASE_MODEL_NAME = "amazon/chronos-2"  # or amazon/chronos-t5-small if using T5 based
VISION_MODEL_NAME = "microsoft/resnet-18" # Lightweight visual backbone
DATA_PATH = "/content/drive/MyDrive/FM_project/dataset/skippd_train_aligned_v13_with_time_features_and_sky_features.parquet"
OUTPUT_DIR = "chronos_vision_fusion_checkpoints"
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 96
BATCH_SIZE = 12 # Adjust based on VRAM (Images take space!)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
COVARIATE_DIM = 16 # Dimension we project images DOWN to (Chronos likes 1-20 covariates)

# --- 1. Define the Fusion Architecture ---
class VisionProjector(nn.Module):
    """
    Learns to translate Visual Embeddings (512 dim) -> Chronos Covariates (16 dim).
    """
    def __init__(self, input_dim=512, output_dim=COVARIATE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ChronosVisionFusion(nn.Module):
    def __init__(self, chronos_model, vision_model_name, projector_dim=COVARIATE_DIM):
        super().__init__()
        self.chronos = chronos_model
        
        # Load Vision Backbone (Frozen)
        # We strip the classification head to get raw features
        print(f"Loading Vision Backbone: {vision_model_name}...")
        self.vision_backbone = AutoModelForImageClassification.from_pretrained(vision_model_name)
        
        # Hack to remove head and get embeddings (Architecture dependent)
        if hasattr(self.vision_backbone, "classifier"):
            self.vision_dim = self.vision_backbone.classifier.in_features
            self.vision_backbone.classifier = nn.Identity()
        elif hasattr(self.vision_backbone, "fc"):
            self.vision_dim = self.vision_backbone.fc.in_features
            self.vision_backbone.fc = nn.Identity()
            
        # Freeze Vision to save VRAM
        for param in self.vision_backbone.parameters():
            param.requires_grad = False
            
        # Learnable Projector
        self.projector = VisionProjector(input_dim=self.vision_dim, output_dim=projector_dim)

    def forward(self, input_ids, attention_mask, target, image_tensors):
        """
        Custom Forward Pass linking Vision -> Chronos
        """
        # 1. Vision Pass (Frozen)
        # Flatten batch and time dimensions for the ResNet: (B*T, C, H, W)
        b, t, c, h, w = image_tensors.shape
        flat_images = image_tensors.view(-1, c, h, w)
        
        with torch.no_grad():
            # Get visual embeddings
            vision_outputs = self.vision_backbone(flat_images).logits # (B*T, 512)
            
        # 2. Projection Pass (Learnable)
        # (B*T, 512) -> (B*T, Cov_Dim)
        projected_covariates = self.projector(vision_outputs)
        
        # Reshape back to sequence: (B, T, Cov_Dim)
        chronos_covariates = projected_covariates.view(b, t, -1)
        
        # 3. Chronos Pass (LoRA)
        # We inject our projected images into the 'feat_dynamic_real' slot
        # Note: Chronos 2 loss calculation is internal
        outputs = self.chronos(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target, # In Chronos training, labels usually = input_ids/target
            feat_dynamic_real=chronos_covariates 
        )
        
        return outputs

# --- 2. Dataset Handling ---
class MultimodalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_processor, context_len, pred_len):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.context_len = context_len
        self.pred_len = pred_len
        
        # Group by Item ID to create valid windows
        self.samples = []
        self._create_windows()

    def _create_windows(self):
        # Sliding window logic
        # For simplicity, we just take the last window per series for this demo
        # In production, iterate over the series to create multiple windows
        for item_id, group in self.df.groupby("item_id"):
            if len(group) < (self.context_len + self.pred_len):
                continue
            
            # Take a window from the end (Training on recent history)
            # You should implement full sliding window here
            window = group.iloc[-(self.context_len + self.pred_len):]
            self.samples.append(window)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_df = self.samples[idx]
        
        # 1. Time Series Data
        target_series = torch.tensor(window_df["pv_value"].values, dtype=torch.float32)
        
        # Chronos Tokenization (Scaling + Binning)
        # Note: We use the tokenizer from the pipeline manually
        # This returns input_ids and attention_mask
        tokenized = self.tokenizer.context_input_transform(
            target_series.unsqueeze(0) # Batch dim required
        )
        
        # 2. Image Data
        # Extract images from the window (list of bytes or paths)
        images = []
        for img_data in window_df["image"]:
            if isinstance(img_data, dict):
                img = Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")
            else:
                # Fallback blank image if missing
                img = Image.new('RGB', (224, 224))
            images.append(img)
            
        # Process images for ResNet
        # returns pixel_values: (Time, 3, 224, 224)
        image_tensors = self.image_processor(images, return_tensors="pt")["pixel_values"]
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": tokenized["input_ids"].squeeze(0), # Self-supervised
            "image_tensors": image_tensors
        }

# --- 3. Main Training Loop ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # A. Load Base Pipeline
    print("Loading Chronos Pipeline...")
    pipeline = BaseChronosPipeline.from_pretrained(
        BASE_MODEL_NAME,
        device_map=device,
        torch_dtype=torch.bfloat16
    )
    
    # B. Configure LoRA for Chronos
    print("Applying LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Adjust based on model arch (T5 vs GPT)
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM" # or SEQ_2_SEQ_LM depending on Chronos variant
    )
    # Wrap the internal model
    pipeline.model = get_peft_model(pipeline.model, peft_config)
    pipeline.model.print_trainable_parameters()
    
    # C. Wrap in Fusion Module
    # This adds the Vision Backbone and Projector
    fusion_model = ChronosVisionFusion(pipeline.model, VISION_MODEL_NAME).to(device)
    
    # D. Prepare Data
    print("Loading Data...")
    df = pd.read_parquet(DATA_PATH)
    # (Optional: Filter df here)
    
    img_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_NAME)
    
    train_ds = MultimodalDataset(
        df, 
        pipeline.tokenizer, 
        img_processor, 
        CONTEXT_LENGTH, 
        PREDICTION_LENGTH
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # E. Optimizer
    # We train: 1. LoRA weights (in pipeline.model) 2. Projector weights
    trainable_params = [p for p in fusion_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    
    # F. Training Loop
    print("Starting Training...")
    fusion_model.train()
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            image_tensors = batch["image_tensors"].to(device) # (B, T, 3, H, W)
            
            # Forward
            outputs = fusion_model(input_ids, attention_mask, labels, image_tensors)
            
            # Loss Calculation
            # Chronos outputs usually contain .loss if labels are provided
            loss = outputs.loss 
            
            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader)}")
        
        # Save Checkpoint
        save_path = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Save LoRA Adapter
        pipeline.model.save_pretrained(os.path.join(save_path, "chronos_adapter"))
        
        # 2. Save Projector (Critical! Without this, the adapter is useless)
        torch.save(fusion_model.projector.state_dict(), os.path.join(save_path, "vision_projector.pt"))
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()