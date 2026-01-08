import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from chronos.chronos2.model import Chronos2Model

class VisionProjector(nn.Module):
    def __init__(self, input_dim=384, output_dim=16, hidden_dim=128):
        """
        Projects high-dimensional visual features (from DinoV2) into 
        low-dimensional 'synthetic covariates' for Chronos.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
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
        use_precomputed_embeddings=False # New flag
    ):
        super().__init__()
        self.covariate_dim = covariate_dim
        self.use_precomputed = use_precomputed_embeddings
        
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
        self.projector = VisionProjector(input_dim=vision_dim, output_dim=covariate_dim)
        
        # 3. The "Brain" (Chronos 2)
        self.chronos = None 

    def forward(
        self, 
        context_tensor,        # (Batch, Time)
        pixel_values,          # (Batch, Total_Time, C, H, W) OR (Batch, Total_Time, Emb_Dim) if precomputed
        group_ids=None,        
        future_target=None     
    ):
        batch_size = pixel_values.shape[0]
        device = context_tensor.device
        
        # --- A. Process Images (The "Eye") ---
        if self.use_precomputed:
            # Input is already embeddings: (Batch, Total_Time, 384)
            # Just flatten relevant dims
            raw_embeddings = pixel_values.reshape(-1, pixel_values.shape[-1]) # (B*T, 384)
        else:
            # Full Vision Path
            _, _, c, h, w = pixel_values.shape
            flat_images = pixel_values.reshape(-1, c, h, w)
            with torch.no_grad():
                vision_outputs = self.vision_backbone(pixel_values=flat_images)
                raw_embeddings = vision_outputs.last_hidden_state[:, 0, :] 
            
        # --- B. Project to Covariates (The "Translator") ---
        visual_flat = self.projector(raw_embeddings)
        
        # Reshape to (Batch, Total_Time, CovariateDim)
        visual_seq = visual_flat.reshape(batch_size, -1, self.covariate_dim)
        
        # Split Visual Sequence into Context and Future parts
        context_len = context_tensor.shape[1]
        
        visual_context = visual_seq[:, :context_len, :]
        
        pred_len = 0
        if future_target is not None:
            pred_len = future_target.shape[1]
            visual_future = visual_seq[:, context_len : context_len+pred_len, :]
        else:
             pass

        # --- C. Expand Batch for Group Attention ---
        pv_context_expanded = context_tensor.unsqueeze(1) 
        
        if future_target is not None:
            pv_future_expanded = future_target.unsqueeze(1)
        
        visual_context_transposed = visual_context.permute(0, 2, 1)
        
        if future_target is not None:
            visual_future_transposed = visual_future.permute(0, 2, 1)
            
        combined_context = torch.cat([pv_context_expanded, visual_context_transposed], dim=1)
        
        if future_target is not None:
            combined_future = torch.cat([pv_future_expanded, visual_future_transposed], dim=1)
            
        num_series = 1 + self.covariate_dim
        
        flat_context = combined_context.reshape(-1, context_len) 
        
        flat_future_target = None
        if future_target is not None:
            flat_future_target = combined_future.reshape(-1, pred_len) 
            
        ids = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, num_series).flatten()
        
        if future_target is not None:
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
