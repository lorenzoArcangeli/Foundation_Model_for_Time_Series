import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import io

class MultimodalDataset(Dataset):
    def __init__(self, df, prediction_length, context_length, image_processor=None, mode="train", use_precomputed=False, stride=1):
        self.df = df
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.image_processor = image_processor
        self.mode = mode
        self.use_precomputed = use_precomputed
        self.stride = stride
        
        # Get unique series IDs
        self.series_ids = self.df['item_id'].unique()
        
        # Group by series_id for faster access
        self.grouped = self.df.groupby('item_id')
        
        # Create samples index
        self.samples = []
        for s_id in self.series_ids:
            series_len = len(self.grouped.get_group(s_id))
            # We need enough data for context + prediction
            if mode == "train":
                # Sliding window with stride
                if series_len > context_length + prediction_length:
                    for i in range(0, series_len - context_length - prediction_length + 1, self.stride):
                         self.samples.append((s_id, i))
            else:
                # Inference: Just take the last window
                if series_len >= context_length:
                    self.samples.append((s_id, series_len - context_length))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s_id, start_idx = self.samples[idx]
        group = self.grouped.get_group(s_id)
        
        # Slicing
        # Total window = Context + Prediction (if training)
        total_len = self.context_length + (self.prediction_length if self.mode == "train" else 0)
        
        window = group.iloc[start_idx : start_idx + total_len]
        
        # 1. Time Series (Target)
        pv_values = torch.tensor(window['pv_value'].values, dtype=torch.float32)
        
        # Split into Context and Target
        context = pv_values[:self.context_length]
        future_target = pv_values[self.context_length:] if self.mode=="train" else None
        
        # 2. Visual Features
        if self.use_precomputed:
             # Load embeddings directly (assuming column 'visual_embedding' is list/array)
             # Expected shape: (Total_Len, 384)
             embeddings = np.stack(window['visual_embedding'].values)
             pixel_values = torch.tensor(embeddings, dtype=torch.float32)
        else:
            # Load Images
            images = []
            for img_data in window['image']:
                try:
                    if isinstance(img_data, dict) and 'bytes' in img_data:
                        image = Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")
                    elif isinstance(img_data, bytes):
                        image = Image.open(io.BytesIO(img_data)).convert("RGB")
                    else:
                        image = Image.new('RGB', (224, 224)) # Black image fallback
                except:
                     image = Image.new('RGB', (224, 224))
                images.append(image)
                
            # Processor
            if self.image_processor:
                # Process strictly returns pixel_values
                encoding = self.image_processor(images, return_tensors="pt")
                pixel_values = encoding.pixel_values # (Total_Len, 3, 224, 224)
            else:
                # Fallback purely for debugging if no processor
                pixel_values = torch.zeros((total_len, 3, 224, 224))
        
        return {
            "context": context,
            "future_target": future_target,
            "pixel_values": pixel_values
        }

def collate_fn(batch):
    # Custom collate because items are Tensors of different lengths (if we had varying lengths), 
    # but here they are fixed.
    # Just stack them.
    
    contexts = torch.stack([item['context'] for item in batch])
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    if batch[0]['future_target'] is not None:
        future_targets = torch.stack([item['future_target'] for item in batch])
    else:
        future_targets = None
        
    return {
        "context": contexts,
        "pixel_values": pixel_values,
        "future_target": future_targets
    }
