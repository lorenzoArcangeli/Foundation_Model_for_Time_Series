import torch
import torch.nn as nn
from transformers import AutoModel
from chronos.chronos2.model import Chronos2Model

class VisionProjector(nn.Module):
    def __init__(self, input_dim=384, output_dim=16, hidden_dim=128, dropout=0.0):
        """
        Projects high-dimensional visual features (from DinoV2) into 
        low-dimensional 'synthetic covariates' for Chronos.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout), # Regularization
            # nn.BatchNorm1d(hidden_dim), # Removed to preserve PCA variance
            # nn.GELU(),                  # Removed to keep linearity like PCA
            nn.Linear(hidden_dim, output_dim),
            # nn.BatchNorm1d(output_dim)  # Removed to avoid destroying variance hierarchy
        )

    def forward(self, x):
        return self.net(x)

class MultimodalChronos(nn.Module):
    def __init__(
        self, 
        chronos_model_name="amazon/chronos-2", 
        vision_model_name="facebook/dinov2-small",
        covariate_dim=16,
        freeze_vision=True,
        use_precomputed_embeddings=True, # New flag
        dropout=0.0, # Dropout for projector
        noise_std=0.0 # Gaussian noise for embeddings during training
    ):
        super().__init__()
        self.covariate_dim = covariate_dim
        self.use_precomputed = use_precomputed_embeddings
        self.noise_std = noise_std
        
        # 1. The "Eye" (Vision Backbone)
        if not self.use_precomputed:
            print(f"Loading Vision Backbone: {vision_model_name}")
            self.vision_backbone = AutoModel.from_pretrained(vision_model_name)
            if freeze_vision:
                for param in self.vision_backbone.parameters():
                    param.requires_grad = False
            vision_dim = self.vision_backbone.config.hidden_size 
        else:
            print("Using precomputed embeddings. Skipping Vision Backbone load.")
            self.vision_backbone = None
            # Assume DinoV2 Small dim if precomputed
            vision_dim = 384 
        
        # 2. The "Translator" (Projector)
        self.projector = VisionProjector(input_dim=vision_dim, output_dim=covariate_dim, dropout=dropout)
        
        # 3. The "Brain" (Chronos 2)
        self.chronos = None 

    def forward(
        self, 
        context_tensor,        # (Batch, Time)
        pixel_values,          # (Batch, Total_Time, C, H, W) OR (Batch, Total_Time, Emb_Dim) if precomputed
        tabular_covariates=None, # (Batch, Total_Time, Num_Covs)
        group_ids=None,        
        future_target=None     
    ):
        batch_size = pixel_values.shape[0]
        device = context_tensor.device
        
        # --- A. Process Images (The "Eye") ---
        # ... (Same as before) ...
        if self.use_precomputed:
            raw_embeddings = pixel_values.reshape(-1, pixel_values.shape[-1]).to(dtype=self.projector.net[0].weight.dtype) 
        else:
            _, _, c, h, w = pixel_values.shape
            flat_images = pixel_values.reshape(-1, c, h, w)
            with torch.no_grad():
                vision_outputs = self.vision_backbone(pixel_values=flat_images)
                raw_embeddings = vision_outputs.last_hidden_state[:, 0, :] 
        
        # --- Robustness: Inject Noise during Training ---
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(raw_embeddings) * self.noise_std
            raw_embeddings = raw_embeddings + noise
            
        # --- B. Project to Covariates (The "Translator") ---
        visual_flat = self.projector(raw_embeddings)
        visual_seq = visual_flat.reshape(batch_size, -1, self.covariate_dim)
        
        # Split Visual Sequence into Context and Future
        context_len = context_tensor.shape[1]
        visual_context = visual_seq[:, :context_len, :]
        
        pred_len = 0
        visual_future = None
        if future_target is not None:
            pred_len = future_target.shape[1]
            visual_future = visual_seq[:, context_len : context_len+pred_len, :]

        # --- C. Process Tabular Covariates ---
        tab_context = None
        tab_future = None
        if tabular_covariates is not None:
            # tabular_covariates is (Batch, Total_Time, Num_Tabular)
            tab_context = tabular_covariates[:, :context_len, :]
            
            if future_target is not None:
                tab_future = tabular_covariates[:, context_len : context_len+pred_len, :]

        # --- D. Expand Batch for Group Attention ---
        # Order: [PV, Tabular, Visual]
        
        pv_context_expanded = context_tensor.unsqueeze(1) # (B, 1, T)
        
        tensors_to_cat = [pv_context_expanded]
        
        if tab_context is not None:
            # (B, T, Num_Tab) -> (B, Num_Tab, T)
            tensors_to_cat.append(tab_context.permute(0, 2, 1))
            
        tensors_to_cat.append(visual_context.permute(0, 2, 1)) # (B, Cov_Dim, T)
        
        combined_context = torch.cat(tensors_to_cat, dim=1)
        
        if future_target is not None:
            pv_future_expanded = future_target.unsqueeze(1)
            future_to_cat = [pv_future_expanded]
            
            if tab_future is not None:
                future_to_cat.append(tab_future.permute(0, 2, 1))
                
            future_to_cat.append(visual_future.permute(0, 2, 1))
            
            combined_future = torch.cat(future_to_cat, dim=1)
            
        # Update num_series to include Tabular dims
        # combined_context is (Batch, Total_Series, T)
        num_series = combined_context.shape[1]
        
        flat_context = combined_context.reshape(-1, context_len) 
        
        flat_future_target = None
        if future_target is not None and pred_len > 0:
            flat_future_target = combined_future.reshape(-1, pred_len) 
            
        ids = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, num_series).flatten()
        
        if future_target is not None and pred_len > 0:
            sample_mask = torch.zeros((num_series, pred_len), dtype=torch.bool, device=device)
            sample_mask[0, :] = True 
            flat_mask = sample_mask.unsqueeze(0).repeat(batch_size, 1, 1).reshape(-1, pred_len)
        else:
            flat_mask = None
            
        # --- D. Forward Pass ---
        output_patch_size = 16 
        num_output_patches = 1
        if future_target is not None:
             import math
             pred_len = future_target.shape[1]
             num_output_patches = math.ceil(pred_len / output_patch_size)
        
        outputs = self.chronos(
            context=flat_context,
            group_ids=ids,
            future_target=flat_future_target,
            future_target_mask=flat_mask,
            num_output_patches=num_output_patches 
        )
        
        return outputs
