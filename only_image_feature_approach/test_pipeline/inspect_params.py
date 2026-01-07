import torch
from visionFusion import VisionChronos2Model, BASE_MODEL_NAME
from transformers import AutoConfig, AutoModel
from peft import PeftModel
import os

# --- Configuration ---
CHECKPOINT_DIR = "chronos_vision_fusion_checkpoints"

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [
        d for d in os.listdir(checkpoint_dir) 
        if d.startswith("checkpoint-epoch-") and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(checkpoint_dir, checkpoints[-1])

CHECKPOINT_PATH = get_latest_checkpoint(CHECKPOINT_DIR)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"Inspecting Checkpoint: {CHECKPOINT_PATH}")
    
    # Load Model
    config = AutoConfig.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model = VisionChronos2Model(config)
    
    # Load Base
    base_model_file = AutoModel.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model.load_state_dict(base_model_file.state_dict(), strict=False)
    
    # Load Projector Weights
    projector_path = os.path.join(CHECKPOINT_PATH, "projector.pt")
    if os.path.exists(projector_path):
        print("Loading Projector...")
        model.vision_projector.load_state_dict(torch.load(projector_path, map_location=DEVICE))
    else:
        print("WARNING: Projector weights not found.")
        return

    # Check output_scale
    scale_val = model.vision_projector.output_scale.item()
    print(f"\n--- Parameter Inspection ---")
    print(f"Initial output_scale: 0.01")
    print(f"Current output_scale: {scale_val:.6f}")
    
    change_pct = ((scale_val - 0.01) / 0.01) * 100
    print(f"Change: {change_pct:.2f}%")
    
    if abs(change_pct) < 1.0:
        print(">>> WARNING: The Projector Scale has barely moved. Learning might be stalled.")
    else:
        print(">>> confirmed: The parameter is updating.")

    # Check Projector Weights Norm
    net = model.vision_projector.net
    last_layer_weight = net[-1].weight
    print(f"Last Layer Weight Mean: {last_layer_weight.mean().item():.6f}")
    print(f"Last Layer Weight Std:  {last_layer_weight.std().item():.6f}")

if __name__ == "__main__":
    main()
