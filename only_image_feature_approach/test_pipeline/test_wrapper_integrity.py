import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoModel
from visionFusion import VisionChronos2Model, DATA_PATH, CONTEXT_LENGTH, PREDICTION_LENGTH, STRIDE, BASE_MODEL_NAME

# Load Checkpoint or Base Weights?
# We want to test the CODE (inference logic), so loading Base Weights is the best control.
# If Base Weights + My Code = Bad, then My Code is broken.

def main():
    print("--- Testing VisionChronos2Model Wrapper Integrity ---")
    
    # 1. Load Data
    df = pd.read_parquet(DATA_PATH)
    # Pick a series
    group = df[df["series_id"] == 0].sort_values("time") # Series 0
    window_size = CONTEXT_LENGTH + PREDICTION_LENGTH
    last_window = group.iloc[-window_size:]
    
    context = torch.tensor(last_window["pv"].values[:CONTEXT_LENGTH], dtype=torch.float32).unsqueeze(0) # (1, 512)
    target = last_window["pv"].values[CONTEXT_LENGTH:]
    
    # Dummy Image
    image_tensor = torch.zeros((1, 3, 224, 224), dtype=torch.float32) 
    
    # 2. Init Model with Base Weights
    print("Loading Base Weights into Custom Wrapper...")
    config = AutoConfig.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model = VisionChronos2Model(config)
    
    base_model = AutoModel.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model.load_state_dict(base_model.state_dict(), strict=False)
    
    # Hack: Zero out the vision projector to mimic "Base Model" behavior perfectly.
    # We suspect the "Random Init" noise (even std=0.01) is destroying the [REG] token signal.
    print("Zeroing out Vision Projector for Control Test...")
    model.vision_projector.output_scale.data.fill_(0.0)
    model.vision_projector.net[-1].weight.data.fill_(0.0)
    model.vision_projector.net[-1].bias.data.fill_(0.0)
    
    model.eval()
    
    # 3. Predict using Custom Method
    print("Running Custom Predict (Vision=0)...")
    with torch.no_grad():
        fc = model.predict(context, image_tensor, PREDICTION_LENGTH)
        # fc shape: (1, quantiles, horizon)
        
    fc_np = fc.numpy()
    median_pred = np.median(fc_np[0], axis=0) # (96,)
    
    # 4. Metrics
    def wmape(y_true, y_pred):
        return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-6) * 100
        
    score = wmape(target, median_pred)
    print(f"Wrapper wMAPE: {score:.2f}%")
    print(f"  Pred Mean: {median_pred.mean():.4f}")
    print(f"  Target Mean: {target.mean():.4f}")
    
    # 5. Plot
    plt.figure()
    plt.plot(target, label="Truth")
    plt.plot(median_pred, label="Wrapper Pred")
    plt.title(f"Wrapper Integrity Test - wMAPE {score:.2f}%")
    plt.legend()
    plt.savefig("test_wrapper.png")
    print("Saved test_wrapper.png")

if __name__ == "__main__":
    main()
