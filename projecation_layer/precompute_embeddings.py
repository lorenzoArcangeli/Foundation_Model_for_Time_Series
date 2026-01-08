import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import pandas as pd
import numpy as np
import io
import os
from tqdm import tqdm

# --- Configuration ---
INPUT_PATH = r"c:\Users\loren\Desktop\D vecchio\UNIVERSITA\MAGISTRALE\SecondYear\FoundationModel\FM_test\project_features\skippd_train_aligned_v13_with_time_features.parquet"
OUTPUT_PATH = r"c:\Users\loren\Desktop\D vecchio\UNIVERSITA\MAGISTRALE\SecondYear\FoundationModel\FM_test\project_features\skippd_train_embeddings.parquet"
VISION_MODEL = "facebook/dinov2-small"
BATCH_SIZE = 32 # Can be larger for inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"Loading dataset from {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)
    
    print(f"Loading Vision Model: {VISION_MODEL} on {DEVICE}...")
    processor = AutoImageProcessor.from_pretrained(VISION_MODEL)
    model = AutoModel.from_pretrained(VISION_MODEL).to(DEVICE)
    model.eval() # Set to eval mode
    
    # Optional: cast to bfloat16 for speed if supported
    model.to(dtype=torch.bfloat16)

    embeddings_list = []
    
    # Buffer for batching
    batch_images = []
    
    print("Starting extraction...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_data = row['image']
        try:
            if isinstance(img_data, dict) and 'bytes' in img_data:
                image = Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")
            elif isinstance(img_data, bytes):
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
            else:
                image = Image.new('RGB', (224, 224))
        except:
             image = Image.new('RGB', (224, 224))
        
        batch_images.append(image)
        
        if len(batch_images) >= BATCH_SIZE or idx == len(df) - 1:
            # Process batch
            inputs = processor(images=batch_images, return_tensors="pt").to(DEVICE)
            # Cast inputs to bfloat16
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Extract CLS token: (Batch, 384)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().float().numpy() 
            
            # Append to list
            embeddings_list.extend([emb for emb in batch_embeddings])
            
            batch_images = []

    # Add to DataFrame
    print(f"Extracted {len(embeddings_list)} embeddings.")
    
    # Verify alignment
    if len(embeddings_list) != len(df):
        print(f"Warning: Embeddings count {len(embeddings_list)} != DataFrame length {len(df)}")
    
    # Create new DataFrame with embeddings instead of raw images
    # We drop the heavy 'image' column and add 'visual_embedding'
    df_new = df.drop(columns=['image'])
    
    # Store as list of floats (easy to save in Parquet)
    df_new['visual_embedding'] = list(embeddings_list)
    
    print(f"Saving to {OUTPUT_PATH}...")
    df_new.to_parquet(OUTPUT_PATH)
    print("Done! 🎉")

if __name__ == "__main__":
    main()
