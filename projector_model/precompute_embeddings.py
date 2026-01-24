import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import pandas as pd
import io
from tqdm import tqdm
from utils import config

def main():
    print(f"Loading dataset from {config.RAW_DATA_PATH}...")
    df = pd.read_parquet(config.RAW_DATA_PATH)
    
    print(f"Loading Vision Model: {config.VISION_MODEL} on {config.DEVICE}...")
    processor = AutoImageProcessor.from_pretrained(config.VISION_MODEL)
    model = AutoModel.from_pretrained(config.VISION_MODEL).to(config.DEVICE)
    model.eval() 
    
    # Cast to bfloat16 for speed
    model.to(dtype=torch.bfloat16)

    embeddings_list = []
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
        
        if len(batch_images) >= config.BATCH_SIZE or idx == len(df) - 1:
            inputs = processor(images=batch_images, return_tensors="pt").to(config.DEVICE)
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Extract CLS token
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().float().numpy() 
            embeddings_list.extend([emb for emb in batch_embeddings])
            
            batch_images = []

    print(f"Extracted {len(embeddings_list)} embeddings.")
    
    if len(embeddings_list) != len(df):
        print(f"Warning: Embeddings count {len(embeddings_list)} != DataFrame length {len(df)}")
    
    # Create new DataFrame with embeddings instead of raw images
    df_new = df.drop(columns=['image'])
    df_new['visual_embedding'] = list(embeddings_list)
    
    print(f"Saving to {config.DATA_PATH}...")
    # Ensure directory exists
    os.makedirs(os.path.dirname(config.DATA_PATH), exist_ok=True)
    df_new.to_parquet(config.DATA_PATH)
    print("Done!")

if __name__ == "__main__":
    main()
