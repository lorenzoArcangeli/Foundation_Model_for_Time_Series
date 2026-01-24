import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from multimodal_chronos import VisionProjector

from utils import config

def main():
    print(f"Loading data from {config.DATA_PATH}...")
    df = pd.read_parquet(config.DATA_PATH)
    
    print("Stacking embeddings...")
    embeddings_list = np.stack(df['visual_embedding'].values) 
    
    X = embeddings_list
    print(f"Data shape: {X.shape}")
    
    # Compute PCA
    print(f"Fitting PCA with {config.COVARIATE_DIM} components...")
    pca = PCA(n_components=config.COVARIATE_DIM)
    pca.fit(X)
    
    components = pca.components_ 
    mean = pca.mean_             
    
    print("PCA Components shape:", components.shape)
    
    # Initialize Projector
    print("Initializing VisionProjector...")
    projector = VisionProjector(input_dim=384, output_dim=config.COVARIATE_DIM, hidden_dim=config.HIDDEN_DIM)
    
    # --- Weight Initialization Strategy ---
    # Strategy: 
    # W1 (Hidden) captures PCA directions.
    # W2 (Output) passes them through.
    
    with torch.no_grad():
        w1 = projector.net[0].weight
        b1 = projector.net[0].bias
        
        w1.zero_()
        b1.zero_()
        
        # Copy PCA components into first COVARIATE_DIM slots
        w1[:config.COVARIATE_DIM] = torch.tensor(components, dtype=torch.float32)
        
        # Bias: Subtract the mean. 
        # Linear layer computes x @ W.T + b
        # PCA computes (x - mean) @ V.T = x @ V.T - mean @ V.T
        # So bias b = - (mean @ V.T)
        
        pca_bias = - np.dot(mean, components.T) 
        b1[:config.COVARIATE_DIM] = torch.tensor(pca_bias, dtype=torch.float32)
        
        w2 = projector.net[2].weight
        b2 = projector.net[2].bias
        
        w2.zero_()
        b2.zero_()
        
        # Make the top-left (16, 16) block Identity
        for i in range(config.COVARIATE_DIM):
            w2[i, i] = 1.0
            
    print("Projector weights initialized with PCA + Identity pass-through.")
    
    # Save
    output_path = os.path.join(config.CHECKPOINT_DIR, "vision_projector_pca_init.pth")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(projector.state_dict(), output_path)
    print(f"Saved initialized projector to {output_path}")

if __name__ == "__main__":
    main()
