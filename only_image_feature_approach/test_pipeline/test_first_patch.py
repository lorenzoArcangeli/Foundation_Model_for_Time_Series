import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoModel
from visionFusion import VisionChronos2Model, DATA_PATH, CONTEXT_LENGTH, STRIDE, BASE_MODEL_NAME
import pandas as pd

def main():
    print("--- Testing First Patch Only (16 steps) ---")
    
    # 1. Data
    df = pd.read_parquet(DATA_PATH)
    group = df[df["series_id"] == 0].sort_values("time")
    
    # Prediction Length = 16 (1 Patch)
    PRED_LEN = 16
    window_size = CONTEXT_LENGTH + PRED_LEN
    last_window = group.iloc[-window_size:] # Get last window that FITS this pred len
    
    context = torch.tensor(last_window["pv"].values[:CONTEXT_LENGTH], dtype=torch.float32).unsqueeze(0)
    target = last_window["pv"].values[CONTEXT_LENGTH:]
    
    # Image (Zeroed)
    img = torch.zeros(1, 3, 224, 224) 
    
    # 2. Model
    config = AutoConfig.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model = VisionChronos2Model(config)
    base_model = AutoModel.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model.load_state_dict(base_model.state_dict(), strict=False)
    
    # Zero Vision
    model.vision_projector.output_scale.data.fill_(0.0)
    model.vision_projector.net[-1].weight.data.fill_(0.0)
    model.vision_projector.net[-1].bias.data.fill_(0.0)
    model.eval()
    
    # 3. Predict (Short Horizon)
    print("Predicting 16 steps...")
    with torch.no_grad():
        fc = model.predict(context, img, PRED_LEN)
        
    fc_np = fc.numpy()
    median_pred = np.median(fc_np[0], axis=0)
    
    # 4. Metrics
    def wmape(y_true, y_pred):
        return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-6) * 100
        
    score = wmape(target, median_pred)
    print(f"First Patch wMAPE: {score:.2f}%")
    print(f"Target Mean: {target.mean()}")
    print(f"Pred Mean: {median_pred.mean()}")
    
    plt.figure()
    plt.plot(target, label="Truth")
    plt.plot(median_pred, label="Pred")
    plt.title(f"First Patch Test - wMAPE {score:.2f}%")
    plt.savefig("test_first_patch.png")

if __name__ == "__main__":
    main()
