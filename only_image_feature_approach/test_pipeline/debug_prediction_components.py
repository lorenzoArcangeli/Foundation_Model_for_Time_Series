import torch
import numpy as np
from visionFusion import VisionChronos2Model, DATA_PATH, CONTEXT_LENGTH, PREDICTION_LENGTH, BASE_MODEL_NAME
import pandas as pd
from transformers import AutoConfig

def main():
    print("--- Debugging Prediction Components ---")
    
    # Data
    df = pd.read_parquet(DATA_PATH)
    group = df[df["series_id"] == 0].sort_values("time")
    window_size = CONTEXT_LENGTH + PREDICTION_LENGTH
    last_window = group.iloc[-window_size:]
    
    # Raw Context (0-25 range usually)
    context_vals = last_window["pv"].values[:CONTEXT_LENGTH]
    print(f"Context Stats | Min: {context_vals.min():.4f}, Max: {context_vals.max():.4f}, Mean: {context_vals.mean():.4f}")
    
    context = torch.tensor(context_vals, dtype=torch.float32).unsqueeze(0)
    img = torch.zeros(1, 3, 224, 224)
    
    # Model
    config = AutoConfig.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model = VisionChronos2Model(config)
    # No weights needed for logic check, just flow
    model.eval()
    
    # Run Encode Manually to get loc/scale
    with torch.no_grad():
        encoder_outputs, loc_scale, _, _ = model.encode(
            context, img, num_output_patches=1
        )
        
    loc, scale = loc_scale
    print(f"\nLoc: {loc.item():.4f}")
    print(f"Scale: {scale.item():.4f}")
    
    # Manual Arcsinh Check
    arcsinh_ctx = np.arcsinh(context_vals)
    print(f"\nManual Arcsinh Mean: {arcsinh_ctx.mean():.4f}")
    print(f"Manual Arcsinh MAE (Abs Mean): {np.mean(np.abs(arcsinh_ctx)):.4f}")
    
    # Check if Loc matches Manual Arcsinh Mean?
    # Chronos might use Mean or Last Value?
    
    # Simulate Predict Flow
    # Assuming quantiles=0 (mean behavior)
    pred_norm = 0.0
    pred_denorm = pred_norm * scale + loc
    # pred_sinh = torch.sinh(pred_denorm)
    
    print(f"\nSimulated Prediction (Quantile=0):")
    print(f"  De-norm (Log Space): {pred_denorm.item():.4f}")
    # print(f"  Sinh (Physical): {pred_sinh.item():.4f}")

    print("\n--- Base Pipeline Internals ---")
    from chronos import BaseChronosPipeline
    pipeline = BaseChronosPipeline.from_pretrained(BASE_MODEL_NAME, device_map="cpu")
    pipeline.model.eval()
    
    with torch.no_grad():
        # Pipeline calls model.encode internally
        # Let's call it manually
        p_out, p_loc_scale, _, _ = pipeline.model.encode(
            context, num_output_patches=1
        )
        p_loc, p_scale = p_loc_scale
        
    print(f"Pipeline Loc: {p_loc.item():.4f}")
    print(f"Pipeline Scale: {p_scale.item():.4f}")
    
    if abs(p_loc.item() - loc.item()) > 1e-4:
        print(">> CRITICAL: Loc Mismatch! Wrapper is doing something different.")
    else:
        print(">> Loc Matches.")
        
    # Check Pipeline Output (before decode?)
    # Pipeline 'predict' does the loop.
    # checking output_patch_embedding result
    p_fc = pipeline.model.output_patch_embedding(p_out[0][:, -1:])
    # p_fc shape (1, 1, 336)
    print(f"Pipeline Logits Mean: {p_fc.mean():.4f}")
    
    # Check what pipeline does with Sinh
    # We can checks source if we could, but let's check config again via pipeline
    print(f"Pipeline Config use_arcsinh: {pipeline.model.chronos_config.use_arcsinh}")

if __name__ == "__main__":
    main()
