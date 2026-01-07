import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoModel
from visionFusion import VisionChronos2Model, DATA_PATH, CONTEXT_LENGTH, PREDICTION_LENGTH, BASE_MODEL_NAME
from einops import rearrange
import pandas as pd

def main():
    print("--- Testing Rearrange Flip ---")
    
    # Data
    df = pd.read_parquet(DATA_PATH)
    group = df[df["series_id"] == 0].sort_values("time")
    window_size = CONTEXT_LENGTH + PREDICTION_LENGTH
    last_window = group.iloc[-window_size:]
    
    context = torch.tensor(last_window["pv"].values[:CONTEXT_LENGTH], dtype=torch.float32).unsqueeze(0)
    target = last_window["pv"].values[CONTEXT_LENGTH:]
    
    # Model (Base Weights, No Vision)
    config = AutoConfig.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model = VisionChronos2Model(config)
    base_model = AutoModel.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model.load_state_dict(base_model.state_dict(), strict=False)
    
    # Zero Vision
    model.vision_projector.output_scale.data.fill_(0.0)
    model.vision_projector.net[-1].weight.data.fill_(0.0)
    model.vision_projector.net[-1].bias.data.fill_(0.0)
    
    model.eval()
    
    # Run Encode
    with torch.no_grad():
        pred_len = PREDICTION_LENGTH
        patch_size = config.chronos_config["output_patch_size"] # Dict access
        num_patches = (pred_len + patch_size - 1) // patch_size
        
        encoder_outputs, loc_scale, _, _ = model.encode(
            context, torch.zeros(1,3,224,224), num_output_patches=num_patches
        )
        
        hidden = encoder_outputs[0]
        forecast_embeds = hidden[:, -num_patches:]
        quantile_preds_logits = model.output_patch_embedding(forecast_embeds)
        
        loc, scale = loc_scale
        loc = loc.unsqueeze(-1)
        scale = scale.unsqueeze(-1)
        
        # --- TEST 1: Original (q p) ---
        print("Testing (q p)...")
        qp_preds = rearrange(quantile_preds_logits, "b n (q p) -> b q (n p)", q=model.num_quantiles, p=patch_size)
        qp_final = qp_preds * scale + loc
        
        # Test 1 Metrics
        qp_med = qp_final[0, 10, :pred_len].numpy() # 10 = Median (0.5) roughly
        wmape_qp = np.sum(np.abs(target - qp_med)) / np.sum(np.abs(target)) * 100
        print(f"Original (q p) wMAPE: {wmape_qp:.2f}%")
        
        # --- TEST 2: Flipped (p q) ---
        print("Testing (p q)...")
        pq_preds = rearrange(quantile_preds_logits, "b n (p q) -> b q (n p)", q=model.num_quantiles, p=patch_size)
        pq_final = pq_preds * scale + loc
        
        # Test 2 Metrics
        pq_med = pq_final[0, 10, :pred_len].numpy()
        wmape_pq = np.sum(np.abs(target - pq_med)) / np.sum(np.abs(target)) * 100
        print(f"Flipped (p q) wMAPE: {wmape_pq:.2f}%")
        
        # Plot
        plt.figure()
        plt.plot(target, label="Truth", color="green")
        plt.plot(qp_med, label=f"Original (q p) {wmape_qp:.0f}%", linestyle="--")
        plt.plot(pq_med, label=f"Flipped (p q) {wmape_pq:.0f}%")
        plt.legend()
        plt.savefig("test_rearrange.png")
        print("Saved test_rearrange.png")

if __name__ == "__main__":
    main()
