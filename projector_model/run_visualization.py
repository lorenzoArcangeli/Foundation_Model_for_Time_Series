import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoImageProcessor
from chronos import BaseChronosPipeline
from peft import PeftModel
from multimodal_chronos import MultimodalChronos
from utils import data_utils
from utils import visualization_utils
from utils import training_utils
from utils import config
CHECKPOINT_EPOCH = 50 # Local config

def load_ft_chronos_model():
    if CHECKPOINT_EPOCH is not None:
        # adapting to step-based or epoch-based folder naming
        actual_checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_step_{CHECKPOINT_EPOCH}")
        print(f"Loading Checkpoint from Step {CHECKPOINT_EPOCH}: {actual_checkpoint_dir}")
    else:
        actual_checkpoint_dir = config.CHECKPOINT_DIR
        print(f"Loading Checkpoint from Root: {actual_checkpoint_dir}")

    model = MultimodalChronos(
        chronos_model_name=config.CHRONOS_MODEL,
        vision_model_name=config.VISION_MODEL,
        covariate_dim=config.COVARIATE_DIM,
        freeze_vision=True,
        use_precomputed_embeddings=True,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )

    # Access pipeline from model
    pipeline = model.pipeline
    print("Base model loaded via MultimodalChronos.")

    # Attach Adapters
    adapter_path = os.path.join(actual_checkpoint_dir, "chronos_lora_adapter")
    pipeline.model = PeftModel.from_pretrained(pipeline.model, adapter_path)
    model.chronos = pipeline.model # Ensure MultimodalChronos uses the PEFT model

    # Load Projector Weights
    projector_path = os.path.join(actual_checkpoint_dir, "vision_projector.pth")
    if os.path.exists(projector_path):
        print(f"Loading Vision Projector from {projector_path}...")
        state_dict = torch.load(projector_path)
        model.projector.load_state_dict(state_dict)
    else:
        print("Warning: Vision Projector weights not found!")

    model.projector.to(dtype=torch.bfloat16)
    model.to(config.DEVICE)
    model.eval()

    return model, pipeline

@torch.no_grad()
def add_projected_features(df, projector):
    """
    Project visual embeddings to covariates.
    """
    print("Projecting visual embeddings to covariates...")

    all_embeddings = np.stack(df['visual_embedding'].values)
    all_tensor = torch.tensor(all_embeddings, dtype=torch.bfloat16, device=config.DEVICE)

    projected_tensor = projector.projector(all_tensor)
    if torch.isnan(projected_tensor).any():
        print("CRITICAL WARNING: Projector output contains NaNs!")
        
    projected_np = projected_tensor.float().cpu().numpy()
    if np.isnan(projected_np).any():
        print("CRITICAL WARNING: Projector numpy output contains NaNs!")

    # Add to DataFrame
    cov_cols = [f"cov_{i}" for i in range(projector.covariate_dim)]
    cov_df = pd.DataFrame(projected_np, columns=cov_cols, index=df.index)
    df_enriched = pd.concat([df, cov_df], axis=1)
    
    if df_enriched[cov_cols].isna().any().any():
         print("CRITICAL WARNING: df_enriched contains NaNs in covariate columns!")

    return df_enriched, cov_cols

def predict(pipeline, train_df, inference_df, test_df, context_length, prediction_length, cov_cols, id_column="item_id", timestamp_column="timestamp", target="pv_value"):
    print(f"Running prediction with {len(cov_cols)} covariates: {cov_cols[:3]}...")
    pred_df = pipeline.predict_df(
        df=train_df,
        future_df=inference_df,
        context_length=context_length,
        prediction_length=prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column=id_column,
        timestamp_column=timestamp_column,
        target=target,
    )
    if 'predictions' in pred_df.columns and pred_df['predictions'].isna().any():
         print("CRITICAL WARNING: predictions column contains NaNs!")
    
    return pred_df

def main():
    # Load Data
    df = data_utils.load_data(config.DATA_PATH)
    
    # Identify initial covariates (excluding array column)
    reserved_columns = ['timestamp', 'item_id', 'pv_value', 'visual_embedding']
    COVARIATE_COLUMNS = [col for col in df.columns if col not in reserved_columns]
    print(f"Initial covariates: {COVARIATE_COLUMNS}")

    # Load Model
    projector, pipeline = load_ft_chronos_model()

    # Project Features
    df_enriched, cov_cols = add_projected_features(df, projector)
    print(f"Added {len(cov_cols)} covariate columns.")

    # Drop raw embedding column for pipeline compatibility
    if 'visual_embedding' in df_enriched.columns:
        df_enriched = df_enriched.drop(columns=['visual_embedding'])
    if 'visual_embedding' in df.columns:
        df = df.drop(columns=['visual_embedding'])

    # Zero-Shot Baseline (Original Data)
    train_df_orig, inference_df_orig, test_df_orig = data_utils.split_ts_dataset(df, config.PREDICTION_LENGTH)

    # Multimodal Prediction
    final_cov = COVARIATE_COLUMNS + cov_cols
    train_df, inference_df, test_df = data_utils.split_ts_dataset(df_enriched, config.PREDICTION_LENGTH)
    
    pred_df_multimodal = predict(pipeline, train_df, inference_df, test_df, config.CONTEXT_LENGTH, config.PREDICTION_LENGTH, final_cov)

    models_to_plot = {
        # "Baseline": pred_df_original,
        "Multimodal": pred_df_multimodal,
    }
    
    # Visualize
    visualization_utils.plot_model_comparison(
        train_df=train_df,
        test_df=test_df,
        model_predictions=models_to_plot,
        plot_history_length=200,   
        prediction_length=config.PREDICTION_LENGTH,
        seasonality=96,
        output_dir=config.RESULTS_DIR
    )

if __name__ == "__main__":
    main()
