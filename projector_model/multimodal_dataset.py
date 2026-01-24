"""
PyTorch datasets for multimodal time series training.

Provides two dataset implementations:
    MultimodalDataset: Deterministic indexing for validation/testing
    RandomMultimodalDataset: Infinite (up to N samples) random sampling for training (matches Chronos pipeline)
"""
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import pandas as pd
import numpy as np
from PIL import Image
import io


class MultimodalDataset(Dataset):
    """
    Deterministic dataset for validation and testing.
    
    In 'train' mode: Creates sliding windows with configurable stride
    In 'validation' mode: Uses only the last window of each series
    """
    def __init__(self, df, prediction_length, context_length, image_processor=None, mode="train", use_precomputed=False, stride=1):
        self.df = df
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.image_processor = image_processor
        self.mode = mode
        self.use_precomputed = use_precomputed
        self.stride = stride
        
        self.series_data = {} 
        grouped = self.df.groupby('item_id')
        self.samples = []
        
        print(f"Preprocessing {mode} dataset into tensors...")
        for s_id, group in grouped:
            pv_values = torch.tensor(group['pv_value'].values, dtype=torch.float32)
            
            embeddings = None
            images_list = None
            
            if use_precomputed:
                emb_list = group['visual_embedding'].values
                embeddings = torch.tensor(np.stack(emb_list), dtype=torch.float32)
                
            else:
                 images_list = group['image'].values 
            
            excluded_cols = ['pv_value', 'visual_embedding', 'image', 'item_id', 'timestamp', 'time', 'series_id']
            cov_cols = [c for c in group.columns if c not in excluded_cols]
            cov_cols.sort()
            
            if cov_cols:
                tabular_covs = torch.tensor(group[cov_cols].values, dtype=torch.float32)
            else:
                tabular_covs = None     

            self.series_data[s_id] = {
                "pv": pv_values,
                "emb": embeddings,
                "img": images_list,
                "tab": tabular_covs,
                "cov_names": cov_cols
            }
            
            series_len = len(group)
            
            if mode == "train":
                if series_len > context_length + prediction_length:
                    for i in range(0, series_len - context_length - prediction_length + 1, self.stride):
                         self.samples.append((s_id, i))
            elif mode == "validation":
                if series_len >= context_length + prediction_length:
                     start_idx = series_len - context_length - prediction_length
                     self.samples.append((s_id, start_idx))
            else:
                if series_len >= context_length:
                    self.samples.append((s_id, series_len - context_length))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s_id, start_idx = self.samples[idx]
        series = self.series_data[s_id]
        
        total_len = self.context_length + (self.prediction_length if self.mode in ["train", "validation"] else 0)
        end_idx = start_idx + total_len
        
        # PV Values
        pv_slice = series["pv"][start_idx : end_idx]
        context = pv_slice[:self.context_length]
        future_target = pv_slice[self.context_length:] if self.mode in ["train", "validation"] else None
        
        # Visual Features
        if self.use_precomputed:
             pixel_values = series["emb"][start_idx : end_idx]
        else:
            images_raw = series["img"][start_idx : end_idx]
            images = []
            for img_data in images_raw:
                try:
                    if isinstance(img_data, dict) and 'bytes' in img_data:
                        image = Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")
                    elif isinstance(img_data, bytes):
                        image = Image.open(io.BytesIO(img_data)).convert("RGB")
                    else:
                        image = Image.new('RGB', (224, 224)) 
                except:
                     image = Image.new('RGB', (224, 224))
                images.append(image)
                
            if self.image_processor:
                encoding = self.image_processor(images, return_tensors="pt")
                pixel_values = encoding.pixel_values 
            else:
                pixel_values = torch.zeros((total_len, 3, 224, 224))
        
        # Tabular Covariates
        tabular_covariates = None
        if series["tab"] is not None:
            tabular_covariates = series["tab"][start_idx : end_idx]
        
        return {
            "context": context,
            "future_target": future_target,
            "pixel_values": pixel_values,
            "tabular_covariates": tabular_covariates
        }

def collate_fn(batch):
    contexts = torch.stack([item['context'] for item in batch])
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    if batch[0]['future_target'] is not None:
        future_targets = torch.stack([item['future_target'] for item in batch])
    else:
        future_targets = None
        
    if batch[0]['tabular_covariates'] is not None:
        tabular_covariates = torch.stack([item['tabular_covariates'] for item in batch])
    else:
        tabular_covariates = None
        
    return {
        "context": contexts,
        "pixel_values": pixel_values,
        "future_target": future_targets,
        "tabular_covariates": tabular_covariates
    }

class RandomMultimodalDataset(IterableDataset):
    """
    Infinite random sampling dataset for training (matches Chronos pipeline behavior).
    
    Sampling strategy:
        1. Select a series with probability proportional to its length
        2. Randomly choose a start position within the valid range
        3. Extract context + prediction windows
    
    Args:
        reserved_end_steps: Number of steps to reserve at the end for validation
    """
    def __init__(self, df, prediction_length, context_length, image_processor=None, use_precomputed=False, reserved_end_steps=0):
        self.df = df
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.image_processor = image_processor
        self.use_precomputed = use_precomputed
        self.reserved_end_steps = reserved_end_steps
        
        self.series_data = {} 
        self.series_ids = []
        self.series_weights = []
        
        dropped_count = 0
        min_required_len = context_length + prediction_length + self.reserved_end_steps
        
        print("Preprocessing dataset for Random Sampling...")
        grouped = self.df.groupby('item_id')

        for s_id, group in grouped:
            total_len = len(group)
            
            if total_len <= min_required_len:
                dropped_count += 1
                continue
                
            self.series_ids.append(s_id)
            
            pv_values = torch.tensor(group['pv_value'].values, dtype=torch.float32)
            
            embeddings = None
            images_list = None
            if use_precomputed:
                emb_list = group['visual_embedding'].values
                embeddings = torch.tensor(np.stack(emb_list), dtype=torch.float32)
            else:
                images_list = group['image'].values
                
            excluded_cols = ['pv_value', 'visual_embedding', 'image', 'item_id', 'timestamp', 'time', 'series_id']
            cov_cols = [c for c in group.columns if c not in excluded_cols]
            cov_cols.sort()
            
            tabular_covs = None
            if cov_cols:
                tabular_covs = torch.tensor(group[cov_cols].values, dtype=torch.float32)
                
            self.series_data[s_id] = {
                "pv": pv_values,
                "emb": embeddings,
                "img": images_list,
                "tab": tabular_covs,
                "len": total_len
            }
            
            weight = total_len - self.context_length - self.prediction_length - self.reserved_end_steps + 1
            if weight < 1: weight = 1 
            self.series_weights.append(weight)

        if dropped_count > 0:
             print(f"WARNING: Dropped {dropped_count} series because they were shorter than {min_required_len} steps.")
            
        self.series_weights = np.array(self.series_weights, dtype=np.float64)
        self.series_probs = self.series_weights / np.sum(self.series_weights)
        
        print(f"RandomDataset ready. Loaded {len(self.series_ids)} valid series.")

    def __iter__(self):
        worker_info = get_worker_info()
        
        while True:
            s_id = np.random.choice(self.series_ids, p=self.series_probs)
            series = self.series_data[s_id]
            
            max_start = series["len"] - self.context_length - self.prediction_length - self.reserved_end_steps
            if max_start <= 0: continue 
            
            start_idx = np.random.randint(0, max_start + 1)
            end_idx = start_idx + self.context_length + self.prediction_length
            
            # PV
            pv_slice = series["pv"][start_idx : end_idx]
            context = pv_slice[:self.context_length]
            future_target = pv_slice[self.context_length:]
            
            # Visual
            if self.use_precomputed:
                pixel_values = series["emb"][start_idx : end_idx]
            else:
                images_raw = series["img"][start_idx : end_idx]
                images = []
                for img_data in images_raw:
                    try:
                        if isinstance(img_data, dict) and 'bytes' in img_data:
                            image = Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")
                        elif isinstance(img_data, bytes):
                            image = Image.open(io.BytesIO(img_data)).convert("RGB")
                        else:
                            image = Image.new('RGB', (224, 224))
                    except:
                        image = Image.new('RGB', (224, 224))
                    images.append(image)
                
                if self.image_processor:
                    encoding = self.image_processor(images, return_tensors="pt")
                    pixel_values = encoding.pixel_values
                else:
                    pixel_values = torch.zeros((len(images), 3, 224, 224))

            # Tabular
            tabular_covariates = None
            if series["tab"] is not None:
                tabular_covariates = series["tab"][start_idx : end_idx]
            
            yield {
                "context": context,
                "future_target": future_target,
                "pixel_values": pixel_values,
                "tabular_covariates": tabular_covariates
            }
