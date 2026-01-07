from chronos import BaseChronosPipeline
import torch
import numpy as np
from visionFusion import BASE_MODEL_NAME

def main():
    print(f"Loading Base Pipeline: {BASE_MODEL_NAME}")
    pipeline = BaseChronosPipeline.from_pretrained(BASE_MODEL_NAME, device_map="cpu")
    model = pipeline.model
    
    print("\n--- Model Config ---")
    print(model.config)
    
    print("\n--- Chronos Config ---")
    try:
        print(model.chronos_config)
    except:
        print("Attribute chronos_config not found directly.")
        
    print("\n--- Checking for Bins / Support ---")
    if hasattr(model, "logits_to_value"):
        print("Found 'logits_to_value'!")
    if hasattr(model, "project_to_support"):
        print("Found 'project_to_support'!")
    
    # Check output head shape
    print("\n--- Output Head Shape ---")
    # input dummy
    dummy = torch.randn(1, 1, model.config.d_model) # 1 patch
    out = model.output_patch_embedding(dummy)
    print(f"Input: (1, 1, {model.config.d_model})")
    print(f"Output: {out.shape}")
    
    # Check if shape matches (quantiles * patch) or (vocab)
    # If vocab, it is categorical.
    
    # Try to find number of quantiles
    nq = getattr(model, "num_quantiles", "Unknown")
    print(f"Num Quantiles: {nq}")
    
    # Try to find tokenizer
    if hasattr(pipeline, "tokenizer"):
        print("Pipeline has tokenizer.")
    else:
        print("Pipeline has NO tokenizer (Chronos 2 is raw values?)")

    import inspect
    print("\n--- Pipeline Predict Source ---")
    try:
        print(inspect.getsource(pipeline.predict))
    except Exception as e:
        print(f"Could not get source: {e}")

if __name__ == "__main__":
    main()
