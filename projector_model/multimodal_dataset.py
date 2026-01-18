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
        
        # --- Optimization: Convert Dataframes to Tensors once ---
        # We need to map (item_id, local_index) -> global_index to slice correctly
        # Or even simpler: we just iterate the series, extract their data, and store a list of Tensors (one per series).
        
        self.series_data = {} # item_id -> { "pv": Tensor, "emb": Tensor }
        
        # Group by series_id 
        grouped = self.df.groupby('item_id')
        
        # Create samples index
        self.samples = []
        
        print("Preprocessing dataset into tensors for speed...")
        for s_id, group in grouped:
            # Extract PV values as Float Tensor
            pv_values = torch.tensor(group['pv_value'].values, dtype=torch.float32)
            
            # Extract Embeddings
            embeddings = None
            images_list = None
            
            if use_precomputed:
                # Fast track: Stack all embeddings for this series at once
                # This is the slow operation, but we do it ONLY ONCE here.
                emb_list = group['visual_embedding'].values
                # Assuming emb_list is array of lists/arrays.
                # np.stack might still be slow if len is huge, but it's done once.
                # Note: np.vstack might be safer if shapes vary, but they shouldn't.
                embeddings = torch.tensor(np.stack(emb_list), dtype=torch.float32)
                
            else:
                 images_list = group['image'].values # Keep as list of bytes/dicts
            
                
            # Extract Tabular Covariates (e.g., time_hour_sin, etc.)
            # Identify columns that are NOT 'pv_value', 'visual_embedding', 'image', 'item_id', 'timestamp', 'time'
            excluded_cols = ['pv_value', 'visual_embedding', 'image', 'item_id', 'timestamp', 'time', 'series_id']
            cov_cols = [c for c in group.columns if c not in excluded_cols]
            
            # Sort cov_cols to ensure deterministic order (e.g., alphabetical)
            # This order MUST match inference/visualization time!
            # In visualization: COVARIATE_COLUMNS (from df minus reserved) + cov_cols (visual)
            # Assuming COVARIATE_COLUMNS are these tabular ones.
            # We should probably pass the list of cov_cols explicitly to be safe, but sorting is a good default.
            cov_cols.sort()
            
            if cov_cols:
                # (Length, Num_Covs)
                tabular_covs = torch.tensor(group[cov_cols].values, dtype=torch.float32)
            else:
                tabular_covs = None     

            # Store in cached dict
            self.series_data[s_id] = {
                "pv": pv_values,
                "emb": embeddings,
                "img": images_list,
                "tab": tabular_covs,
                "cov_names": cov_cols
            }
            
            series_len = len(group)
            
            # Generate Sample Indices
            if mode == "train":
                if series_len > context_length + prediction_length:
                    for i in range(0, series_len - context_length - prediction_length + 1, self.stride):
                         self.samples.append((s_id, i))
            elif mode == "validation":
                # Only take the LAST valid window
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
        
        # Slice ranges
        total_len = self.context_length + (self.prediction_length if self.mode in ["train", "validation"] else 0)
        end_idx = start_idx + total_len
        
        # DEBUG: Log sampling info (100% of samples)
        with open("random_sample_log.txt", "a") as f:
             f.write(f"[{self.mode.upper()}] Series: {s_id} | Start: {start_idx} | End: {end_idx} (Len: {len(series['pv'])})\n")
        
        # 1. PV Values (Instant Tensor Slice)
        pv_slice = series["pv"][start_idx : end_idx]
        
        context = pv_slice[:self.context_length]
        future_target = pv_slice[self.context_length:] if self.mode in ["train", "validation"] else None
        
        if self.use_precomputed:
             # Instant Tensor Slice
             pixel_values = series["emb"][start_idx : end_idx]
        else:
            # Load Images (Legacy Slow Path)
            # We access the raw list of bytes/dicts stored in 'img'
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
                
            # Processor
            if self.image_processor:
                encoding = self.image_processor(images, return_tensors="pt")
                pixel_values = encoding.pixel_values 
            else:
                pixel_values = torch.zeros((total_len, 3, 224, 224))
        
        # 3. Tabular Covariates
        tabular_covariates = None
        if series["tab"] is not None:
            # (Total_Len, Num_Covs)
            tabular_covariates = series["tab"][start_idx : end_idx]
        
        return {
            "context": context,
            "future_target": future_target,
            "pixel_values": pixel_values,
            "tabular_covariates": tabular_covariates
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

class RandomMultimodalDataset(torch.utils.data.IterableDataset):
    """
    Dataset that implements 'Infinite Random Sampling' logic similar to ChronosPipeline.fit().
    Instead of pre-computing a list of all possible windows (which is huge with stride=1),
    it randomly selects a series and a random valid start time on-the-fly.
    """
    def __init__(self, df, prediction_length, context_length, image_processor=None, use_precomputed=False, reserved_end_steps=0):
        self.df = df
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.image_processor = image_processor
        self.use_precomputed = use_precomputed
        self.reserved_end_steps = reserved_end_steps
        
        # Preprocess Data into Tensors
        self.series_data = {} # item_id -> { "pv": Tensor, "emb": Tensor, ... }
        self.series_ids = []
        self.series_weights = []
        
        dropped_count = 0
        min_required_len = context_length + prediction_length + self.reserved_end_steps
        
        print("Preprocessing dataset for Random Sampling...")
        grouped = self.df.groupby('item_id')

        for s_id, group in grouped:
            # Skip series that are too short
            total_len = len(group)
            
            if total_len <= min_required_len:
                dropped_count += 1
                continue
                
            self.series_ids.append(s_id)
            
            # Extract PV
            pv_values = torch.tensor(group['pv_value'].values, dtype=torch.float32)
            
            # Extract Embeddings/Images
            embeddings = None
            images_list = None
            if use_precomputed:
                emb_list = group['visual_embedding'].values
                embeddings = torch.tensor(np.stack(emb_list), dtype=torch.float32)
            else:
                images_list = group['image'].values
                
            # Extract Tabular
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
            
            # Calculate Valid Sampling Weight
            # The number of valid start indices is (max_start - 0 + 1)
            # max_start = total_len - context - prediction - reserved
            weight = total_len - self.context_length - self.prediction_length - self.reserved_end_steps + 1
            if weight < 1: weight = 1 
            self.series_weights.append(weight)

        if dropped_count > 0:
             print(f"WARNING: Dropped {dropped_count} series because they were shorter than {min_required_len} steps.")
            
        # Normalize Probabilities
        self.series_weights = np.array(self.series_weights, dtype=np.float64)
        self.series_probs = self.series_weights / np.sum(self.series_weights)
        
        print(f"RandomDataset ready. Loaded {len(self.series_ids)} valid series.")
        print(f"Sampling Weights: Min={self.series_probs.min():.6f}, Max={self.series_probs.max():.6f}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # Ensure reproducibility per worker if needed, but for now just rely on numpy random state
        
        while True:
            # 1. Pick a Random Series (Weighted by Length)
            s_id = np.random.choice(self.series_ids, p=self.series_probs)
            series = self.series_data[s_id]
            
            # 2. Pick a Random Valid Start Index
            # valid range: [0, total_len - context - prediction - reserved]
            max_start = series["len"] - self.context_length - self.prediction_length - self.reserved_end_steps
            if max_start <= 0: continue # Should have been filtered, but safety check
            
            start_idx = np.random.randint(0, max_start + 1)
            end_idx = start_idx + self.context_length + self.prediction_length
            
            # DEBUG: Log sampling info to file (100% of samples)
            with open("random_sample_log.txt", "a") as f:
                 f.write(f"Series: {s_id} | Start: {start_idx} | End: {end_idx} (Len: {series['len']})\n")
            
            # 3. Slice Data (Identical logic to Map-Style __getitem__)
            # PV
            pv_slice = series["pv"][start_idx : end_idx]
            context = pv_slice[:self.context_length]
            future_target = pv_slice[self.context_length:]
            
            # Visual
            if self.use_precomputed:
                pixel_values = series["emb"][start_idx : end_idx]
            else:
                # Handling raw images on the fly if needed (slower)
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
