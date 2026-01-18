import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from multimodal_chronos import VisionProjector

# --- Configuration ---
DATA_PATH = r"c:\Users\loren\Desktop\D vecchio\UNIVERSITA\MAGISTRALE\SecondYear\FoundationModel\FM_test\project_features\datasets\skippd_train_embeddings.parquet"
OUTPUT_PATH = r"c:\Users\loren\Desktop\D vecchio\UNIVERSITA\MAGISTRALE\SecondYear\FoundationModel\FM_test\project_features\multimodal_infinite_checkpoints\vision_projector_pca_init.pth"
COVARIATE_DIM = 16 
HIDDEN_DIM = 128

def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Extract all embeddings
    # df['visual_embedding'] is column of arrays/lists
    print("Stacking embeddings...")
    embeddings_list = np.stack(df['visual_embedding'].values) # (N_Samples, 384)
    # Note: These are 'per timestamp' embeddings if original data was per-timestamp. 
    # If the dataframe has time steps, we have N rows.
    
    X = embeddings_list
    print(f"Data shape: {X.shape}")
    
    # Compute PCA
    print(f"Fitting PCA with {COVARIATE_DIM} components...")
    pca = PCA(n_components=COVARIATE_DIM)
    pca.fit(X)
    
    components = pca.components_ # (n_components, n_features) -> (16, 384)
    mean = pca.mean_             # (384,)
    
    print("PCA Components shape:", components.shape)
    
    # Initialize Projector
    print("Initializing VisionProjector...")
    projector = VisionProjector(input_dim=384, output_dim=COVARIATE_DIM, hidden_dim=HIDDEN_DIM)
    
    # --- Weight Initialization Strategy ---
    # We want Projector(x) ~ PCA(x)
    # PCA(x) = (x - mean) @ V.T
    # Projector uses MLP: Linear -> BN -> GELU -> Linear -> BN
    
    # Strategy: 
    # W1 (Hidden) captures PCA directions.
    # W2 (Output) passes them through.
    
    with torch.no_grad():
        # 1. First Linear Layer (384 -> 128)
        # We place the 16 PCA components into the first 16 rows of W1
        # The rest can be random or zero.
        
        # W1 is (128, 384)
        w1 = projector.net[0].weight
        b1 = projector.net[0].bias
        
        # Zero out first
        w1.zero_()
        b1.zero_()
        
        # Copy PCA components into first COVARIATE_DIM slots
        # components is (16, 384). w1[:16] is (16, 384).
        w1[:COVARIATE_DIM] = torch.tensor(components, dtype=torch.float32)
        
        # Bias: We need to subtract the mean. 
        # Linear layer computes x @ W.T + b
        # PCA computes (x - mean) @ V.T = x @ V.T - mean @ V.T
        # So bias b = - (mean @ V.T)
        
        pca_bias = - np.dot(mean, components.T) # (16,)
        b1[:COVARIATE_DIM] = torch.tensor(pca_bias, dtype=torch.float32)
        
        # 2. (Removed BN/GELU)
        
        # 3. Second Linear Layer (128 -> 16)
        # We want to select the first 16 dimensions and pass them to output.
        # W2 is the SECOND Linear layer now (index 2) because Dropout is at index 1.
        # Indices:
        # 0: Linear(384->128)
        # 1: Dropout
        # 2: Linear(128->16)
        
        # Note: If you commented them out in the class, they don't exist in the list.
        # So indices are 0, 1, 2.
        
        w2 = projector.net[2].weight
        b2 = projector.net[2].bias
        
        w2.zero_()
        b2.zero_()
        
        # Make the top-left (16, 16) block Identity
        # So it takes the 16 PCA features created in Layer 1 and passes them.
        for i in range(COVARIATE_DIM):
            w2[i, i] = 1.0
            
        # 4. (Removed BN2)
        
    print("Projector weights initialized with PCA + Identity pass-through.")
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(projector.state_dict(), OUTPUT_PATH)
    print(f"Saved initialized projector to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
