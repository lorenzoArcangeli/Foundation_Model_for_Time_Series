import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor, AutoConfig
from chronos.chronos2.model import Chronos2Model, Chronos2EncoderOutput, Chronos2Output
from peft import LoraConfig, get_peft_model
import pandas as pd
import numpy as np
from PIL import Image
import io
import os
from tqdm import tqdm
from einops import rearrange, repeat
import copy
from typing import cast

# --- Configuration ---
BASE_MODEL_NAME = "amazon/chronos-2" 
VISION_MODEL_NAME = "facebook/dinov2-small"
DATA_PATH = r"c:\Users\loren\Desktop\D vecchio\UNIVERSITA\MAGISTRALE\SecondYear\FoundationModel\FM_test\skippd_train_aligned_v13_with_time_features.parquet"
OUTPUT_DIR = "chronos_vision_fusion_checkpoints"
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 96
BATCH_SIZE = 12
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
STRIDE = 12

# --- 1. Define Fusion Architecture ---

class VisionProjector(nn.Module):
    """Projects Visual Embeddings (384) to Chronos Model Dimension."""
    def __init__(self, input_dim=384, output_dim=None):
        super().__init__()
        assert output_dim is not None, "Must provide Chronos model_dim"
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2), # Expand first
            nn.GELU(),
            nn.LayerNorm(output_dim * 2), # Stabilize intermediate
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim) # Project to d_model
        )
        
        # Initialize the last layer with very small weights to match Chronos embedding scale (~0.03)
        # This prevents the vision magnitude (usually ~1-5) from dominating the attention
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01) # Slightly larger base
        nn.init.zeros_(self.net[-1].bias)
        
        # Explicit Learnable Scale
        self.output_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        # x: (batch, input_dim)
        out = self.net(x) * self.output_scale
        return out.unsqueeze(1) # Return (batch, 1, d_model) for sequence concatenation

class VisionChronos2Model(Chronos2Model):
    """
    Subclass of Chronos2Model that injects visual embeddings into the encoder.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # 1. Initialize Vision Backbone
        print(f"Loading Vision Backbone: {VISION_MODEL_NAME}...")
        self.vision_backbone = AutoModel.from_pretrained(VISION_MODEL_NAME)
        self.vision_dim = 384
        
        # Freeze Vision Backbone
        for param in self.vision_backbone.parameters():
            param.requires_grad = False
            
        # 2. Initialize Projector (Maps Vision Dim -> Chronos d_model)
        self.vision_projector = VisionProjector(
            input_dim=self.vision_dim, 
            output_dim=config.d_model
        )

    def encode(
        self,
        context: torch.Tensor,
        image_tensors: torch.Tensor, # [NEW] Image Input
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
        future_target: torch.Tensor | None = None,
        future_target_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ):
        """
        Modified encode method to inject visual embeddings.
        """
        # Validate input (standard)
        self._validate_input(
            context=context,
            context_mask=context_mask,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            group_ids=group_ids,
            num_output_patches=num_output_patches,
            future_target=future_target,
            future_target_mask=future_target_mask,
        )

        batch_size = context.shape[0]
        
        # [CRITICAL FIX] Apply input scaling (Arcsinh) 
        # The model expects transformed inputs. Pipeline usually handles this via Dataset.
        # Since we feed raw tensors, we must transform them here.
        if self.chronos_config.use_arcsinh:
            context = torch.arcsinh(context)
            if future_target is not None:
                future_target = torch.arcsinh(future_target)
        
        # --- A. Process Context (Time Series) ---
        patched_context, attention_mask, loc_scale = self._prepare_patched_context(
            context=context, context_mask=context_mask
        )
        # input_embeds shape: (batch, num_context_patches, d_model)
        input_embeds: torch.Tensor = self.input_patch_embedding(patched_context)
        
        # [CRITICAL UPDATE: Sum Fusion]
        # Instead of inserting a new token (which breaks relative positions),
        # We ADD the vision embedding to the [REG] token.
        
        # 1. Get Vision Proj
        # image_tensors shape: (batch, 3, H, W)
        with torch.no_grad():
            vision_out = self.vision_backbone(image_tensors)
            vision_embeds_raw = vision_out.last_hidden_state[:, 0, :]
            
        vision_embeds_proj = self.vision_projector(vision_embeds_raw) # (batch, 1, d_model)
        vision_embeds_proj = vision_embeds_proj.to(input_embeds.dtype)

        # 2. Add to REG Token (if enabled)
        if self.chronos_config.use_reg_token:
            reg_input_ids = torch.full((batch_size, 1), self.config.reg_token_id, device=input_embeds.device)
            reg_embeds = self.shared(reg_input_ids)
            
            # --- FUSION: Summation ---
            # Enriches the "Transition Token" with visual context without changing sequence length
            reg_embeds = reg_embeds + vision_embeds_proj
            
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [attention_mask.to(self.dtype), torch.ones_like(reg_input_ids).to(self.dtype)], dim=-1
            )
        else:
            # Fallback for models without REG: Prepend (Early Fusion) but this is risky for relative pos
            # Ideally Chronos-2 always has REG.
            input_embeds = torch.cat([vision_embeds_proj, input_embeds], dim=1)
            vision_mask = torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # --- C. Process Future ---
        patched_future, patched_future_covariates_mask = self._prepare_patched_future(
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            loc_scale=loc_scale,
            num_output_patches=num_output_patches,
            batch_size=batch_size,
        )
        
        future_attention_mask = torch.ones(batch_size, num_output_patches, dtype=self.dtype, device=self.device)
        future_embeds: torch.Tensor = self.input_patch_embedding(patched_future)

        # Concatenate everything
        input_embeds = torch.cat([input_embeds, future_embeds], dim=-2)
        attention_mask = torch.cat([attention_mask, future_attention_mask], dim=-1)

        if group_ids is None:
            group_ids = torch.arange(batch_size, dtype=torch.long, device=self.device)

        # --- D. Run Encoder ---
        encoder_outputs = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            group_ids=group_ids,
            output_attentions=output_attentions,
        )
        
        # Return tuple matching standard output + extra info
        # Note: num_context_patches needs to match the tensor size for correct slicing later
        # We added 1 vision token, so we increment this logic?
        # Chronos output slicing: forecast_embeds = hidden_states[:, -num_output_patches:]
        # So "num_context_patches" returned here is just informational for the caller, 
        # but let's keep it accurate to the "time series" patches
        num_context_patches = input_embeds.shape[1] - num_output_patches # approx
        
        return encoder_outputs, loc_scale, patched_future_covariates_mask, num_context_patches

    def forward(
        self,
        context: torch.Tensor,
        image_tensors: torch.Tensor,
        future_target: torch.Tensor | None = None,
        # Default args for compatibility
        context_mask=None, group_ids=None, future_covariates=None, 
        future_covariates_mask=None, num_output_patches=1, future_target_mask=None, output_attentions=False
    ) -> Chronos2Output:
        
        batch_size = context.shape[0]
        
        # Logic to infer num_output_patches if future_target is provided
        if future_target is not None:
             # Calculate required patches
             required_len = future_target.shape[-1]
             patch_size = self.chronos_config.output_patch_size
             num_output_patches = (required_len + patch_size - 1) // patch_size
        
        encoder_outputs, loc_scale, patched_future_covariates_mask, _ = self.encode(
            context=context,
            image_tensors=image_tensors,
            context_mask=context_mask,
            group_ids=group_ids,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            num_output_patches=num_output_patches,
            future_target=future_target,
            future_target_mask=future_target_mask,
            output_attentions=output_attentions,
        )
        
        hidden_states = encoder_outputs[0]
        
        # Prediction Head
        # slice last num_output_patches
        forecast_embeds = hidden_states[:, -num_output_patches:]
        quantile_preds = self.output_patch_embedding(forecast_embeds)
        
        # Reshape to (batch, quantiles, horizon)
        quantile_preds = rearrange(
            quantile_preds,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=self.num_quantiles,
            p=self.chronos_config.output_patch_size,
        )
        
        # Compute Loss
        loss = None
        if future_target is not None:
            loss = self._compute_loss(
                quantile_preds=quantile_preds,
                future_target=future_target,
                future_target_mask=future_target_mask,
                patched_future_covariates_mask=patched_future_covariates_mask,
                loc_scale=loc_scale,
                num_output_patches=num_output_patches,
            )

        # Output
        return Chronos2Output(
            loss=loss,
            quantile_preds=quantile_preds
        )

    def predict(
        self,
        context: torch.Tensor,
        image_tensors: torch.Tensor,
        prediction_length: int | None = None,
        num_samples: int | None = None, # Not used for quantile model effectively, but kept for API
    ):
        """
        Inference method that returns DE-NORMALIZED predictions.
        """
        batch_size = context.shape[0]
        if prediction_length is None:
            # Fallback or error? Let's default to class config or error
            prediction_length = 96 # Hardcoded default or fetch from config if possible
            
        patch_size = self.chronos_config.output_patch_size
        # [CRITICAL UPDATE: Autoregressive Loop]
        # Chronos predicts one patch at a time. We must loop.
        
        predictions = []
        current_context = context
        remaining_steps = prediction_length
        
        while remaining_steps > 0:
            # 1. Encode & Predict Next Patch
            # Note: We must re-encode (or manage KV cache if optimized)
            # For simplicity, we re-encode the growing context.
            encoder_outputs, loc_scale, _, _ = self.encode(
                context=current_context,
                image_tensors=image_tensors,
                num_output_patches=1
            )
            
            # 2. Get Last Hidden State (Prediction for Next Patch)
            hidden = encoder_outputs[0] # (B, Seq, Dim)
            next_patch_embed = hidden[:, -1:, :] # Last token (REG) holds the prediction
            
            # 3. Project to Quantiles
            patch_logits = self.output_patch_embedding(next_patch_embed) # (B, 1, 336)
            
            # 4. Save Prediction (Logits)
            predictions.append(patch_logits)
            
            # 5. Decode to Values for Autoregression
            # We need to feed the *value* back as context.
            # Sample or Mean? Base Pipeline uses 'unrolled_quantiles'? 
            # Or does it feed embeddings? 
            # Base Pipeline: `_autoregressive_unroll...` feeds `prediction` back.
            # `_predict_step` returns *samples* or *logits*?
            # It returns *prediction*.
            
            # Let's Decode the logits to Sample/Mean to update context.
            # Base Chronos uses "Median" (0.5) or sampled?
            # Usually for stability we feed the Median.
            
            # Shape: (B, 1, Q*P) -> (B, Q, P)
            p_size = self.chronos_config.output_patch_size
            n_q = self.num_quantiles
            
            # Reshape
            patch_preds = patch_logits.view(batch_size, 1, n_q, p_size)
            patch_preds = patch_preds.permute(0, 2, 1, 3).squeeze(2) # (B, Q, P)
            
            # Decode scale
            loc, scale = loc_scale
            loc = loc.unsqueeze(-1)
            scale = scale.unsqueeze(-1)
            
            patch_values = patch_preds * scale + loc
            if self.chronos_config.use_arcsinh:
                patch_values = torch.sinh(patch_values)
                
            # Take Median (Index 10 for 21 quantiles)
            median_idx = 10
            next_patch_values = patch_values[:, median_idx, :] # (B, P)
            
            # 6. Update Context
            # We must apply arcsinh again because 'encode' expects RAW (or we fix 'encode'?)
            # Wait, 'encode' now applies Arcsinh to 'context'.
            # So we should feed RAW values to 'encode' loop?
            # Yes, 'current_context' is raw.
            current_context = torch.cat([current_context, next_patch_values], dim=1)
            
            remaining_steps -= p_size
            
            # Safety: Crop later
            
        # Concat all logits
        quantile_preds_logits = torch.cat(predictions, dim=1) # (B, NumPatches, 336)
        
        # --- Final Decode ---
        p_size = self.chronos_config.output_patch_size
        n_q = self.num_quantiles
        
        # Reshape: (Batch, NumPatches, Quantiles * PatchSize) -> (Batch, Quantiles, TotalTime)
        # Note: output_patch_embedding outputs [q1_p1, q1_p2... q2_p1, q2_p2] flattened?
        # Inspect model structure said (1, 1, 336). 21 * 16 = 336.
        # So it's (Q * P).
        # We need to confirm the order: Q first or P first?
        # test_rearrange_flip showed Original (q p) was same WMAPE as Flipped.
        # But logically: usually [Q1...Qn] for P1, then P2? 
        # No, the head is (D -> Q*P). 
        # Standard Chronos 2: (B, N, Q*P) -> (B, N, Q, P)
        
        quantile_preds = quantile_preds_logits.view(batch_size, -1, n_q, p_size)
        quantile_preds = quantile_preds.permute(0, 2, 1, 3) # (B, Q, N, P)
        quantile_preds = quantile_preds.reshape(batch_size, n_q, -1) # (B, Q, Horizon)
        
        # Scale
        # Note: loc/scale are from the *Original* Context? Or updated?
        # Usually we use the initial loc/scale for the whole window unless rolling.
        # We'll use the initial one.
        
        # Re-calc initial loc/scale for correct shape
        # [CRITICAL] Ensure we use Arcsinh context if config enabled, to match 'encode' behavior
        ctx_for_scale = context
        if self.chronos_config.use_arcsinh:
            ctx_for_scale = torch.arcsinh(context)
            
        _, _, loc_scale = self._prepare_patched_context(ctx_for_scale, torch.ones_like(ctx_for_scale, dtype=torch.bool))
        loc, scale = loc_scale
        loc = loc.unsqueeze(-1)
        scale = scale.unsqueeze(-1)
        
        prediction = quantile_preds * scale + loc
        
        if self.chronos_config.use_arcsinh:
            prediction = torch.sinh(prediction)
        
        # Crop
        prediction = prediction[..., :prediction_length]
        
        return prediction

# --- 2. Dataset Handling ---

class MultimodalDataset(Dataset):
    def __init__(self, dataframe, image_processor, context_len, pred_len, stride=STRIDE):
        self.df = dataframe
        self.image_processor = image_processor
        self.context_len = context_len
        self.pred_len = pred_len
        self.stride = stride
        self.samples = []
        self._create_windows()

    def _create_windows(self):
        window_size = self.context_len + self.pred_len
        for item_id, group in tqdm(self.df.groupby("series_id"), desc="Generating Windows"):
            # Ensure group is sorted by time if needed, assuming aligned
            num_samples = len(group)
            if num_samples < window_size:
                continue
            
            for i in range(0, num_samples - window_size + 1, self.stride):
                window = group.iloc[i : i + window_size]
                self.samples.append(window)
        print(f"Generated {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_df = self.samples[idx]
        
        # Get raw values
        series_values = window_df["pv"].values.astype(np.float32)
        
        # Split into Context and Target
        context = series_values[:self.context_len]
        target = series_values[self.context_len:]
        
        # Get Image (Last image in context? Or first? Or middle?)
        # Strategy: Use the image at the END of the context window (most recent visual state)
        # Assumes 'image' column is dict
        img_data = window_df["image"].iloc[self.context_len - 1] 
        
        if isinstance(img_data, dict) and 'bytes' in img_data:
            img = Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")
        else:
             img = Image.new('RGB', (224, 224)) # Helper
             
        # Process Image
        image_tensor = self.image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
        
        return {
            "context": torch.tensor(context),
            "future_target": torch.tensor(target),
            "image_tensors": image_tensor
        }

# --- 3. Main Training Loop ---

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Config & Model
    print("Loading Base Config...")
    config = AutoConfig.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    print("Initializing VisionChronos2Model...")
    # Initialize our custom class with the standard config
    model = VisionChronos2Model(config)
    
    # Load Pretrained Weights (Partial)
    print("Loading Pretrained Weights...")
    base_model = AutoModel.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    # Copy state dict (excluding vision parts which are not in base)
    model.load_state_dict(base_model.state_dict(), strict=False)
    del base_model # Free memory
    
    model.to(device)
    
    # 2. Setup LoRA
    print("Applying LoRA to Chronos Parts...")
    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    # Target only the 'encoder' part of our model for LoRA
    # We can apply LoRA to the whole model, it will find the linear layers in encoder
    # But we want to ensure we don't LoRA the frozen vision backbone (it's frozen anyway)
    # and we want the Projector to be fully trainable (not LoRA)
    
    model = get_peft_model(model, peft_config)
    
    # Make sure Projector is Trainable (get_peft_model might freeze non-detected modules depending on config)
    # Explicitly unfreeze projector
    for param in model.base_model.model.vision_projector.parameters():
        param.requires_grad = True
        
    model.print_trainable_parameters()
    
    # 3. Data
    print("Loading Dataset...")
    df = pd.read_parquet(DATA_PATH)
    img_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_NAME)
    
    train_ds = MultimodalDataset(df, img_processor, CONTEXT_LENGTH, PREDICTION_LENGTH, stride=STRIDE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4. Optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # 5. Train
    print("Starting Training...")
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            context = batch["context"].to(device)
            future_target = batch["future_target"].to(device)
            image_tensors = batch["image_tensors"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                context=context,
                image_tensors=image_tensors,
                future_target=future_target
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Save
        save_path = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
        model.save_pretrained(save_path) # Saves LoRA adapter + Base
        # Manually save Projector state dict if needed, typically PEFT saves adapters
        # But 'vision_projector' is a new unique module. PEFT might NOT save it automatically if it's considered 'base'
        # Best to safe-guard:
        torch.save(model.base_model.model.vision_projector.state_dict(), os.path.join(save_path, "projector.pt"))

if __name__ == "__main__":
    main()