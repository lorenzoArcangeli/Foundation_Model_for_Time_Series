import torch
import numpy as np
from transformers import AutoConfig, AutoModel
from visionFusion import VisionChronos2Model, BASE_MODEL_NAME, VisionProjector
from chronos import BaseChronosPipeline

def main():
    print("--- Debugging Internals: Pipeline vs Wrapper ---")
    
    # 1. Setup Data
    context_data = torch.randn(1, 512) # Random context
    image_tensor = torch.zeros(1, 3, 224, 224) 
    
    # 2. Load Pipeline (Reference)
    print(f"Loading Pipeline: {BASE_MODEL_NAME}")
    pipeline = BaseChronosPipeline.from_pretrained(BASE_MODEL_NAME, device_map="cpu", torch_dtype=torch.float32)
    base_model = pipeline.model
    # Ensure Eval
    base_model.eval()
    
    # 3. Load Wrapper (Student)
    print("Loading Wrapper...")
    config = AutoConfig.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    wrapper = VisionChronos2Model(config)
    wrapper.load_state_dict(base_model.state_dict(), strict=False)
    
    # Hack: Force Wrapper's VisionProjector to output ZEROS
    # to ensure identical behavior to Base (which has no vision)
    # Since we use Sum Fusion (REG + Vision), if Vision=0, then Wrapper should == Base.
    wrapper.vision_projector.output_scale.data.fill_(0.0)
    wrapper.vision_projector.net[-1].weight.data.fill_(0.0)
    wrapper.vision_projector.net[-1].bias.data.fill_(0.0)
    
    wrapper.eval()
    
    # 4. Compare 'encode' steps (Simulated)
    # We can't easily hook into 'encode' without modifying code, 
    # but we can look at the Final Output of "predict" logic steps.
    
    # --- Base Pipeline Prediction ---
    # The pipeline uses model.encode() internally too.
    # Let's call model.encode manually on both.
    
    print("\n--- Comparing Encode Output ---")
    with torch.no_grad():
        # Pipeline model
        # Needs args: context, num_output_patches
        # Base model 'encode' signature might differ from ours if we overrode it?
        # VisionChronos2Model overrides 'encode'.
        # Base implementation is in 'chronos.chronos2.model'.
        
        # We need to match arguments.
        # wrapper.encode(context, image_tensors, num_output_patches=...)
        # base_model.encode(context, num_output_patches=...)
        
        pred_len = 96
        patch_size = config.chronos_config["output_patch_size"] # Dict access
        num_patches = (pred_len + patch_size - 1) // patch_size
        
        # Base
        base_out, base_loc, _, _ = base_model.encode(
            context=context_data,
            num_output_patches=num_patches
        )
        base_hidden = base_out[0]
        
        # Wrapper
        wrapper_out, wrapper_loc, _, _ = wrapper.encode(
            context=context_data,
            image_tensors=image_tensor,
            num_output_patches=num_patches
        )
        wrapper_hidden = wrapper_out[0]
        
        diff_hidden = (base_hidden - wrapper_hidden).abs().mean().item()
        diff_loc = (base_loc[0] - wrapper_loc[0]).abs().mean().item() # loc
        diff_scale = (base_loc[1] - wrapper_loc[1]).abs().mean().item() # scale
        
        print(f"Hidden State Diff: {diff_hidden:.6f}")
        print(f"Loc Diff: {diff_loc:.6f}")
        print(f"Scale Diff: {diff_scale:.6f}")
        
        if diff_hidden > 1e-5:
            print(">> HUGE DIVERGENCE in Encoder Output!")
            # Check Input Embeddings size
            print(f"Base Hidden Shape: {base_hidden.shape}")
            print(f"Wrapper Hidden Shape: {wrapper_hidden.shape}")
        
        # --- Compare Decode/Predict ---
        print("\n--- Comparing Final Prediction ---")
        
        # Base Pipeline Logic (Manual)
        base_fc_embeds = base_hidden[:, -num_patches:]
        base_quantiles = base_model.output_patch_embedding(base_fc_embeds)
        # Reshape omitted for brevity, checking tensor values first
        
        # Wrapper Pipeline Logic (from predict)
        wrapper_fc_embeds = wrapper_hidden[:, -num_patches:]
        wrapper_quantiles = wrapper.output_patch_embedding(wrapper_fc_embeds)
        
        diff_quantiles = (base_quantiles - wrapper_quantiles).abs().mean().item()
        print(f"Quantiles logits Diff: {diff_quantiles:.6f}")
        
if __name__ == "__main__":
    main()
