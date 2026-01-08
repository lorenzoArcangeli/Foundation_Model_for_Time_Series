import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import pandas as pd
import io
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = "/content/drive/MyDrive/FM_project/dataset"
INPUT_PATH = os.path.join(BASE_DIR, "skippd_train_aligned_v13_with_time_features.parquet")
OUTPUT_PATH = os.path.join(BASE_DIR, "skippd_train_aligned_v13_with_time_features_and_sky_features.parquet")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "feature_extractors") # To save PCA/Scaler

MODEL_NAME = "facebook/dinov2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
N_COMPONENTS = 10

# Ensure save directory exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# --- 1. Define the Extractor Class ---
class SkyFeatureExtractor:
    def __init__(self, model_name=MODEL_NAME, device=DEVICE):
        print(f"Loading {model_name} on {device}...")
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_features(self, images):
        if not images: return np.array([])
        # Preprocess
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Get CLS token (index 0)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# --- 2. Processing Loop ---
def process_full_dataset(input_path, extractor, batch_size=32):
    print(f"Reading dataset from {input_path}...")
    df = pd.read_parquet(input_path)

    all_features = []
    batch_images = []

    print("Starting feature extraction...")
    # Iterate through the DataFrame
    for raw_data in tqdm(df["image"], desc="Extracting Features"):
        try:
            # Handle dictionary format (standard for this dataset)
            if isinstance(raw_data, dict):
                img_bytes = raw_data.get('bytes')
                if img_bytes:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    batch_images.append(img)
                else:
                    # Fallback if bytes are missing: create a black image to keep alignment
                    batch_images.append(Image.new('RGB', (224, 224)))
            else:
                 batch_images.append(Image.new('RGB', (224, 224)))

            # If batch is full, run inference
            if len(batch_images) >= batch_size:
                features = extractor.extract_features(batch_images)
                all_features.append(features)
                batch_images = [] # Clear memory

        except Exception as e:
            print(f"Error processing image: {e}")
            # Add dummy zeros to maintain row count alignment
            all_features.append(np.zeros((1, 384)))

    # Process leftovers
    if batch_images:
        features = extractor.extract_features(batch_images)
        all_features.append(features)

    # Stack into one big array
    return df, np.vstack(all_features)

# --- 3. Execution Main ---

# A. Extract Raw Features
extractor = SkyFeatureExtractor()
df, raw_features = process_full_dataset(INPUT_PATH, extractor, batch_size=BATCH_SIZE)
print(f"Raw features extracted. Shape: {raw_features.shape}")

# B. Normalize (StandardScaler)
print("Fitting Standard Scaler...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(raw_features)

# C. Reduce Dimensions (PCA)
print(f"Fitting PCA (n={N_COMPONENTS})...")
pca = PCA(n_components=N_COMPONENTS)
pca_features = pca.fit_transform(features_scaled)

# SAFETY CHECK: Do these 10 features actually matter?
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
print(f"Explained Variance by 10 components: {cumulative_variance[-1]:.2%}")

# Visualization of feature importance (Optional but recommended)
# This tells you if Feature 9 and 10 are actually useless noise
import matplotlib.pyplot as plt
plt.plot(cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Is 10 too many? Check the elbow.')
plt.grid(True)
plt.show()

explained_variance = np.sum(pca.explained_variance_ratio_)
print(f"✅ Explained Variance: {explained_variance:.2%}")

# D. Add to DataFrame
print("Merging new columns...")
feature_cols = [f"sky_feature_{i}" for i in range(N_COMPONENTS)]
df_features = pd.DataFrame(pca_features, columns=feature_cols, index=df.index)

# Concatenate along columns
df_final = pd.concat([df, df_features], axis=1)

# E. Save Everything
print(f"Saving new dataset to {OUTPUT_PATH}...")
df_final.to_parquet(OUTPUT_PATH)

# Save the models (CRITICAL for processing your Test Set later)
with open(os.path.join(MODEL_SAVE_PATH, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODEL_SAVE_PATH, "pca.pkl"), "wb") as f:
    pickle.dump(pca, f)

print("Done! 🎉")
print(df_final[feature_cols].head())