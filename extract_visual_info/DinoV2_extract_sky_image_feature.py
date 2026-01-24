import os
import io
import pickle
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

BASE_DIR = "/content/drive/MyDrive/FM_project/dataset"
INPUT_PATH = os.path.join(BASE_DIR, "skippd_train_aligned_v13_with_time_features.parquet")
OUTPUT_PATH = os.path.join(BASE_DIR, "skippd_train_aligned_v13_with_time_features_and_sky_features.parquet")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "feature_extractors")
MODEL_NAME = "facebook/dinov2-small"
BATCH_SIZE = 32
N_COMPONENTS = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SkyFeatureExtractor:
    """
    Extracts visual features from images using a pre-trained DinoV2 model.
    """
    def __init__(self, model_name=MODEL_NAME, device=DEVICE):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_features(self, images):
        """
        Extracts features for a batch of images.
        Returns the CLS token embedding.
        
        Args:
            images (list): List of PIL images.
            
        Returns:
            np.ndarray: Extracted features (CLS token).
        """
        if not images:
            return np.array([])
        
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Return CLS token (index 0)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def process_dataset_features(df, extractor, batch_size=32):
    """
    Iterates through the dataframe, processing images in batches.
    Handles missing images by creating black placeholders.

    Args:
        df (pd.DataFrame): Dataframe containing 'image' column.
        extractor (SkyFeatureExtractor): Initialized feature extractor.
        batch_size (int): Size of the batch for inference.
    
    Returns:
        np.ndarray: Stacked array of extracted features.
    """
    all_features = []
    batch_images = []

    for raw_data in tqdm(df["image"], desc="Extracting Features"):
        try:
            # Handle dictionary format containing bytes
            if isinstance(raw_data, dict):
                img_bytes = raw_data.get('bytes')
                if img_bytes:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    batch_images.append(img)
                else:
                    batch_images.append(Image.new('RGB', (224, 224)))
            else:
                batch_images.append(Image.new('RGB', (224, 224)))

            if len(batch_images) >= batch_size:
                features = extractor.extract_features(batch_images)
                all_features.append(features)
                batch_images = []

        except Exception as e:
            print(f"Error processing image: {e}")
            # Append zero vector as fallback (384 is dinov2-small dim)
            all_features.append(np.zeros((1, 384)))

    # Process remaining images
    if batch_images:
        features = extractor.extract_features(batch_images)
        all_features.append(features)

    return np.vstack(all_features) if all_features else np.array([])

def save_variance_plot(cumulative_variance, save_dir):
    """
    Saves the PCA explained variance plot.
    """
    plt.figure()
    plt.plot(cumulative_variance)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plot_path = os.path.join(save_dir, "pca_variance.png")
    plt.savefig(plot_path)
    plt.close()

def main():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Load Data
    print(f"Loading dataset from {INPUT_PATH}...")
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file {INPUT_PATH} not found.")
        pass
    try:
        df = pd.read_parquet(INPUT_PATH)
    except Exception as e:
        print(f"Could not read parquet file (expected if running locally with colab paths): {e}")
        return

    # Extract Features
    extractor = SkyFeatureExtractor()
    raw_features = process_dataset_features(df, extractor, batch_size=BATCH_SIZE)
    print(f"Extracted features shape: {raw_features.shape}")

    # Normalize Features
    print("Normalizing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(raw_features)

    # Apply PCA
    print(f"Applying PCA (n_components={N_COMPONENTS})...")
    pca = PCA(n_components=N_COMPONENTS)
    pca_features = pca.fit_transform(features_scaled)

    # Calculate and log variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print(f"Explained Variance by {N_COMPONENTS} components: {cumulative_variance[-1]:.2%}")
    save_variance_plot(cumulative_variance, MODEL_SAVE_DIR)

    # Merge and Save
    print("Merging and saving dataset...")
    feature_cols = [f"sky_feature_{i}" for i in range(N_COMPONENTS)]
    df_features = pd.DataFrame(pca_features, columns=feature_cols, index=df.index)
    df_final = pd.concat([df, df_features], axis=1)

    df_final.to_parquet(OUTPUT_PATH)
    
    # Save feature extractors for inference
    with open(os.path.join(MODEL_SAVE_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_SAVE_DIR, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)

    print(f"Processing complete. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()