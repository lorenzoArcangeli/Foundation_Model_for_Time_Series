import torch
import pandas as pd
import numpy as np
from transformers import AutoConfig, AutoImageProcessor, AutoModel
from peft import PeftModel
from visionFusion import VisionChronos2Model, MultimodalDataset, BASE_MODEL_NAME, VISION_MODEL_NAME, DATA_PATH, CONTEXT_LENGTH, PREDICTION_LENGTH, STRIDE
import os

# --- Configuration ---
CHECKPOINT_PATH = "chronos_vision_fusion_checkpoints/checkpoint-epoch-10" # Adjust to latest
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_for_debug(checkpoint_path):
    print("Loading Config...")
    config = AutoConfig.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model = VisionChronos2Model(config)
    
    print("Loading Weights...")
    base_model_file = AutoModel.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model.load_state_dict(base_model_file.state_dict(), strict=False)
    
    projector_path = os.path.join(checkpoint_path, "projector.pt")
    if os.path.exists(projector_path):
        print(f"Loading Projector: {projector_path}")
        model.vision_projector.load_state_dict(torch.load(projector_path, map_location=DEVICE))
        
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.to(DEVICE)
    model.eval()
    return model

def analyze_embeddings(model, dataset):
    sample = dataset[0]
    context = sample["context"].to(DEVICE).unsqueeze(0)
    image_tensors = sample["image_tensors"].to(DEVICE).unsqueeze(0)
    
    # We need to hook or manually run the 'encode' parts to see intermediate values
    # Let's subclass or monkey-patch, but since we have the code, easier to just copy-paste the logic 
    # OR, better: Add a temporary print inside the class? 
    # NO, we can't edit the class file easily without losing the user's focus.
    # Let's write a "Shadow Encode" function here that replicates the logic exactly.
    
    print("\n--- DEBUGGING EMBEDDINGS ---")
    
    # 1. Vision Forward
    with torch.no_grad():
        vision_out = model.base_model.model.vision_backbone(image_tensors)
        vision_embeds_raw = vision_out.last_hidden_state[:, 0, :]
        vision_embeds_proj = model.base_model.model.vision_projector(vision_embeds_raw)
        vision_embeds_proj = vision_embeds_proj.unsqueeze(1)
        
        print(f"Vision Raw Mean: {vision_embeds_raw.mean().item():.4f}, Std: {vision_embeds_raw.std().item():.4f}, Norm: {vision_embeds_raw.norm().item():.4f}")
        print(f"Vision Proj Mean: {vision_embeds_proj.mean().item():.4f}, Std: {vision_embeds_proj.std().item():.4f}, Norm: {vision_embeds_proj.norm().item():.4f}")
        
    # 2. Chronos Context Forward
    # We need to access the 'input_patch_embedding' layer
    # model is PeftModel -> base_model -> model -> input_patch_embedding
    chronos_model = model.base_model.model
    
    patched_context, _, _ = chronos_model._prepare_patched_context(context, None)
    input_embeds = chronos_model.input_patch_embedding(patched_context)
    
    print(f"Context Emb Mean: {input_embeds.mean().item():.4f}, Std: {input_embeds.std().item():.4f}, Norm: {input_embeds.norm().item():.4f}")
    
    # Check Ratio
    ratio = vision_embeds_proj.abs().mean() / input_embeds.abs().mean()
    print(f"Magnitude Ratio (Vision/Context): {ratio.item():.2f}")
    
    if ratio > 5.0 or ratio < 0.2:
        print(">>> CRITICAL WARNING: Embedding scales are mismatched! Attention will be broken.")
    else:
        print(">>> Embedding scales look comparable.")

def main():
    model = load_model_for_debug(CHECKPOINT_PATH)
    df = pd.read_parquet(DATA_PATH)
    img_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_NAME)
    ds = MultimodalDataset(df, img_processor, CONTEXT_LENGTH, PREDICTION_LENGTH, stride=STRIDE)
    
    analyze_embeddings(model, ds)

if __name__ == "__main__":
    main()
