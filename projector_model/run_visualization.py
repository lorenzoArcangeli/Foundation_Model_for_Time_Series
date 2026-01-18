import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from transformers import AutoModel
from chronos import BaseChronosPipeline
from peft import PeftModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Import our custom modules
from multimodal_chronos import MultimodalChronos, VisionProjector
# We don't need MultimodalDataset anymore for this approach!

# --- Configuration ---
BASE_DIR = "/content/drive/MyDrive/FM_project/dataset"
DATA_PATH = os.path.join(BASE_DIR,"skippd_train_embeddings.parquet")
CHECKPOINT_DIR = "/content/multimodal_checkpoints_dim_16_batch_16"
CHECKPOINT_EPOCH = 50 # Set to None to use the root dir, or an integer (e.g., 5, 10) to use a specific epoch checkpoint
VISION_MODEL = "facebook/dinov2-small"
CHRONOS_MODEL = "amazon/chronos-2"
CONTEXT_LENGTH = 4096
PREDICTION_LENGTH = 96
prediction_length = PREDICTION_LENGTH
COVARIATE_DIM = 16
target = "pv_value"
id_column = "item_id"
timestamp_column = "timestamp"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_ft_chronos_model():
    # Construct Payload Path based on Epoch
    if CHECKPOINT_EPOCH is not None:
        #actual_checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{CHECKPOINT_EPOCH}")
        actual_checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{CHECKPOINT_EPOCH}")
        print(f"Loading Checkpoint from Epoch {CHECKPOINT_EPOCH}: {actual_checkpoint_dir}")
    else:
        actual_checkpoint_dir = CHECKPOINT_DIR
        print(f"Loading Checkpoint from Root: {actual_checkpoint_dir}")

    model = MultimodalChronos(
        chronos_model_name=CHRONOS_MODEL,
        vision_model_name=VISION_MODEL,
        covariate_dim=COVARIATE_DIM,
        freeze_vision=True,
        use_precomputed_embeddings=True
    )

    # 2. Load the Base Model
    pipeline = BaseChronosPipeline.from_pretrained(
        CHRONOS_MODEL,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )

    print("Base model loaded.")

    # 3. Attach your LoRA/DoRA Adapter
    # The adapter is saved in nested folders inside the checkpoint dir usually, or directly.
    # Based on your zip file, it seems the structure is 'checkpoint_epoch_X/chronos_lora_adapter'
    adapter_path = os.path.join(actual_checkpoint_dir, "chronos_lora_adapter")

    pipeline.model = PeftModel.from_pretrained(
        pipeline.model,
        adapter_path
    )

    # Load Projector Weights
    projector_path = os.path.join(actual_checkpoint_dir, "vision_projector.pth")
    if os.path.exists(projector_path):
        print(f"Loading Vision Projector from {projector_path}...")
        state_dict = torch.load(projector_path)
        model.projector.load_state_dict(state_dict)
    else:
        print("Warning: Vision Projector weights not found!")

    # Cast Projector and Move
    model.projector.to(dtype=torch.bfloat16)
    model.to(DEVICE)
    model.eval()

    return model, pipeline


# --- 2. Feature Engineering ---
@torch.no_grad()
def add_projected_features(df, projector):
    """
    Takes the dataframe with 384-dim embeddings.
    Runs the projector.
    Adds 16 columns 'cov_0'...'cov_15' to the dataframe.
    """
    print("Projecting visual embeddings to covariates...")

    all_embeddings = np.stack(df['visual_embedding'].values) # (N, 384)
    all_tensor = torch.tensor(all_embeddings, dtype=torch.bfloat16, device=DEVICE)


    # Project
    # (N, 16)
    projected_tensor = projector.projector(all_tensor)
    if torch.isnan(projected_tensor).any():
        print("CRITICAL WARNING: Projector output contains NaNs!")
        
    projected_np = projected_tensor.float().cpu().numpy()
    if np.isnan(projected_np).any():
        print("CRITICAL WARNING: Projector numpy output contains NaNs!")

    # Add to DataFrame
    cov_cols = [f"cov_{i}" for i in range(projector.covariate_dim)]

    # Create a DataFrame of new cols
    cov_df = pd.DataFrame(projected_np, columns=cov_cols, index=df.index)

    # Concatenate
    df_enriched = pd.concat([df, cov_df], axis=1)
    
    # Debug: Check enriched df
    if df_enriched[cov_cols].isna().any().any():
         print("CRITICAL WARNING: df_enriched contains NaNs in covariate columns!")

    return df_enriched, cov_cols

def predict(pipeline, train_df, inference_df, test_df, context_length, prediction_length, cov_cols):
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
    # Debug: Check predictions for NaNs
    if 'predictions' in pred_df.columns and pred_df['predictions'].isna().any():
         print("CRITICAL WARNING: predictions column contains NaNs!")
    
    return pred_df



# --- Metric & Plotting (Same as before) ---
def calculate_item_mase(y_true, y_pred, y_history, seasonality=96):
    """Calculates MASE for a single item."""
    # Mean Absolute Error of the forecast
    forecast_mae = np.mean(np.abs(y_true - y_pred))

    # Mean Absolute Error of the naive baseline on history
    # (comparing t with t-seasonality)
    if len(y_history) <= seasonality:
        return np.inf # Not enough history for seasonality

    naive_errors = np.abs(y_history[seasonality:] - y_history[:-seasonality])
    naive_mae = np.mean(naive_errors)

    if naive_mae == 0:
        return np.inf

    return forecast_mae / naive_mae

def calculate_item_mape(y_true, y_pred, epsilon=1e-10):
    """
    Calculates Mean Absolute Percentage Error (MAPE).

    Args:
        epsilon (float): Small value to avoid division by zero.
                         Alternatively, you can mask values where y_true == 0.
    """
    # Option A: Add epsilon to avoid zero division (simplest)
    # mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    # Option B: Filter out zeros (standard for PV data)
    # We only compute MAPE when actual production > 0
    mask = y_true > epsilon
    if np.sum(mask) == 0:
        return np.nan # No valid data points (e.g., essentially night time)

    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return mape

def calculate_item_wmape(y_true, y_pred):
    """
    Calculates Weighted Mean Absolute Percentage Error (wMAPE).
    This is preferred for intermittent series (like Solar) where y_true can be ~0.
    """
    total_abs_error = np.sum(np.abs(y_true - y_pred))
    total_actuals = np.sum(np.abs(y_true))

    if total_actuals == 0:
        return np.inf

    return (total_abs_error / total_actuals) * 100

def calculate_item_wql(y_true, quantile_preds, quantiles):
    """
    Calculates Weighted Quantile Loss (WQL) for a single item.

    Args:
        y_true (np.array): Ground truth values (T,)
        quantile_preds (dict): Dictionary mapping quantile (float) to prediction array (T,)
        quantiles (list): List of quantiles to evaluate [0.1, 0.5, 0.9]
    """
    total_loss = 0
    total_abs_target = np.sum(np.abs(y_true))

    if total_abs_target == 0:
        return np.inf

    for q in quantiles:
        y_pred_q = quantile_preds[q]
        # Quantile Loss: 2 * (y - y_pred) * (q if y > y_pred else q-1)
        # Note: The factor '2' is common in some definitions (like GluonTS) to make it comparable to absolute error for q=0.5
        # Standard QL: (1-q)|y-y_hat| if y < y_hat, q|y-y_hat| if y >= y_hat

        errors = y_true - y_pred_q
        loss = np.maximum(q * errors, (q - 1) * errors)

        # GluonTS / WQL definition typically sums this over time
        total_loss += np.sum(2 * loss)

    # Average over number of quantiles
    wql = total_loss / (len(quantiles) * total_abs_target)
    return wql

def calculate_item_sql(y_true, quantile_preds, quantiles, y_history, seasonality=96):
    """
    Calculates Scaled Quantile Loss (SQL) for a single item.
    Denominator is the naive error (same as MASE).
    """
    total_loss = 0

    # Calculate Denominator (Naive Error)
    if len(y_history) <= seasonality:
        return np.inf

    naive_errors = np.abs(y_history[seasonality:] - y_history[:-seasonality])
    naive_mae = np.mean(naive_errors)

    if naive_mae == 0:
        return np.inf

    for q in quantiles:
        y_pred_q = quantile_preds[q]
        errors = y_true - y_pred_q
        loss = np.maximum(q * errors, (q - 1) * errors)

        total_loss += np.mean(2 * loss) # Mean over prediction buffer

    # Average over quantiles and divide by naive mae
    sql = total_loss / (len(quantiles) * naive_mae)
    return sql


# 2. Main Plotting Function
def plot_model_comparison(train_df, test_df, model_predictions,
                          plot_history_length=200, prediction_length=96, seasonality=96):

    # A. Setup Data
    # Concatenate train and test to get the full timeline for plotting context
    full_data = pd.concat([train_df, test_df]).sort_values(['item_id', 'timestamp'])

    item_ids = sorted(full_data['item_id'].unique())
    num_plots = len(item_ids)

    # Safety check for huge datasets
    if num_plots > 20:
        print(f"Warning: Plotting {num_plots} series will create a very large image.")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Blue, Orange, Green, Red

    # B. Initialize Figure
    fig, axs = plt.subplots(num_plots, 1, figsize=(15, 6 * num_plots), sharex=False)
    if num_plots == 1: axs = [axs] # Ensure axs is always iterable

    print(f"Generating plots for {num_plots} items...")

    for i, item_id in enumerate(item_ids):
        ax = axs[i]

        # --- C. Prepare Series Data ---
        # Filter data for this specific item
        item_full_data = full_data[full_data['item_id'] == item_id].set_index('timestamp')
        item_test_data = test_df[test_df['item_id'] == item_id].set_index('timestamp')
        item_train_data = train_df[train_df['item_id'] == item_id].set_index('timestamp')

        # 1. History Context (for plotting)
        # We grab the last 'plot_history_length' points from training + the prediction window
        history_context = item_full_data.iloc[-(plot_history_length + prediction_length):]

        # 2. Ground Truth (The actual future)
        ground_truth_future = item_test_data['pv_value']

        # 3. History for Metric (for MASE denominator)
        history_for_metric = item_train_data['pv_value'].values

        # --- D. Plot Background ---
        # Plot the actual values (history + future)
        ax.plot(history_context.index, history_context['pv_value'],
                label='Actual Ground Truth', color='black', linewidth=2, alpha=0.6)

        # Highlight the Forecast Window
        if len(ground_truth_future) > 0:
            cutoff_date = ground_truth_future.index[0]
            ax.axvspan(cutoff_date, ground_truth_future.index[-1], color='gray', alpha=0.1, label="Forecast Window")
            ax.axvline(x=cutoff_date, color='black', linestyle=':', linewidth=1)

        # --- E. Plot Each Model ---
        for idx, (model_name, pred_df_all) in enumerate(model_predictions.items()):
            # Filter predictions for this item
            item_preds = pred_df_all[pred_df_all['item_id'] == item_id].set_index('timestamp')

            if item_preds.empty:
                continue

            # 1. Calculate Metrics
            try:
                # Align lengths (take last N points if necessary)
                y_pred_median = item_preds['predictions'].values[-len(ground_truth_future):]
                y_true = ground_truth_future.values[-len(y_pred_median):]

                # MASE
                mase_score = calculate_item_mase(
                    y_true=y_true,
                    y_pred=y_pred_median,
                    y_history=history_for_metric,
                    seasonality=seasonality
                )
                # MAPE
                mape_score = calculate_item_mape(y_true, y_pred_median)
                # wMAPE
                wmape = calculate_item_wmape(y_true, y_pred_median)

                # WQL & SQL
                # Check if we have quantiles
                quantiles_to_check = [0.1, 0.5, 0.9]
                quantile_preds_dict = {}
                has_quantiles = True

                # Check keys in dataframe columns (strings: '0.1', '0.5', '0.9')
                for q in quantiles_to_check:
                    q_str = str(q)
                    if q_str in item_preds.columns:
                        quantile_preds_dict[q] = item_preds[q_str].values[-len(ground_truth_future):]
                    else:
                        has_quantiles = False
                        # Fallback for 0.5 if missing (use median prediction)
                        if q == 0.5:
                            quantile_preds_dict[0.5] = y_pred_median

                if has_quantiles:
                    wql_score = calculate_item_wql(y_true, quantile_preds_dict, quantiles_to_check)
                    sql_score = calculate_item_sql(y_true, quantile_preds_dict, quantiles_to_check, history_for_metric, seasonality)

                    metrics_label = (f"MASE:{mase_score:.2f} MAPE:{mape_score:.0f}% wMAPE:{wmape:.1f}% "
                                     f"WQL:{wql_score:.3f} SQL:{sql_score:.3f}")
                else:
                    metrics_label = f"MASE:{mase_score:.2f} MAPE:{mape_score:.0f}% wMAPE:{wmape:.1f}%"

            except Exception as e:
                print(f"Error calculating metrics for {model_name} on item {item_id}: {e}")
                metrics_label = "Metrics: Error"

            # 2. Plot Predictions
            color = colors[idx % len(colors)]
            label_text = f'{model_name}\n({metrics_label})'

            # Plot Median (prediction)
            ax.plot(item_preds.index, item_preds['predictions'],
                    label=label_text, color=color, linewidth=2, linestyle='--')

            # Plot Quantiles (if available)
            if '0.1' in item_preds.columns and '0.9' in item_preds.columns:
                ax.fill_between(
                    item_preds.index,
                    item_preds['0.1'],
                    item_preds['0.9'],
                    color=color, alpha=0.15
                )

        # --- F. Final Formatting ---
        ax.set_title(f"Item {item_id}: Forecast Comparison", fontsize=14, fontweight='bold')
        ax.set_ylabel("PV Value")
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    #If figuere already present save with an incremental id
    if os.path.exists(f"forecast_comparison_{item_id}.png"):
        i=1
        while os.path.exists(f"forecast_comparison_{item_id}_{i}.png"):
            i+=1
        plt.savefig(f"forecast_comparison_{item_id}_{i}.png")
    else:
        plt.savefig(f"forecast_comparison_{item_id}.png")

    plt.tight_layout()
    plt.show()

# --- 3. Execute ---





def load_and_prepare_data():
    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)

    column_mapping = {
        "time": "timestamp",
        "series_id": "item_id",
        "pv": "pv_value"
    }

    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    if 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time'])
        if df['timestamp'].dt.tz is not None:
             df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    df = df.sort_values(['item_id', 'timestamp'])

    initial_cov=reserved_columns = ['timestamp', 'item_id', 'pv_value']
    # Filter out visual_embedding from covariates list as it is an array column
    COVARIATE_COLUMNS = [col for col in df.columns if col not in reserved_columns and col != 'visual_embedding']

    print(f" Automatically identified {len(COVARIATE_COLUMNS)} covariates: {COVARIATE_COLUMNS}")
    # We don't strictly require missing covariates check here since we just derived them from df.columns
    # but keeping it safe.

    return df, COVARIATE_COLUMNS


def split_ts_dataset(df):
    # --- UPDATED LOGIC FOR MULTIPLE SERIES ---

    # 4. Filter out series that are too short
    # We need at least prediction_length + 1 data point to have a training set
    item_counts = df.groupby('item_id').size()
    valid_items = item_counts[item_counts > PREDICTION_LENGTH].index

    if len(valid_items) < len(item_counts):
        print(f"Dropping {len(item_counts) - len(valid_items)} series that are too short.")
        df = df[df['item_id'].isin(valid_items)].copy()

    # 5. Sort by item_id AND timestamp (Critical for correct splitting)
    df = df.sort_values(['item_id', 'timestamp']).reset_index(drop=True)

    # 6. Global Split
    print(f"Splitting data for {len(valid_items)} time series...")

    # Inference/Test: Grab the last PREDICTION_LENGTH rows for EACH item_id
    test_df = df.groupby('item_id').tail(PREDICTION_LENGTH).copy()

    # Train: Drop the rows that belong to test_df
    # (Since we reset_index above, the indices are unique and safe to use for dropping)
    train_df = df.drop(test_df.index).copy()

    # 7. Create Inference input (Drop target)
    inference_df = test_df.copy()
    if 'pv_value' in inference_df.columns:
        inference_df = inference_df.drop(columns=['pv_value'])

    print(f"Train shape: {train_df.shape}")
    print(f"Inference/Test shape: {inference_df.shape}")

    return train_df, inference_df, test_df


def main():

    df, COVARIATE_COLUMNS = load_and_prepare_data()


    # 2. Load Model
    projector, pipeline = load_ft_chronos_model()

    # 3. Project Features (This matches user's request to "create dataset with feature")
    df_enriched, cov_cols = add_projected_features(df, projector)
    print(f"Added {len(cov_cols)} covariate columns.")

    # FIX: Drop the raw embedding column (numpy arrays) because standard pipeline functions
    # (like predict_df) try to hash/factorize columns and fail on arrays.
    if 'visual_embedding' in df_enriched.columns:
        df_enriched = df_enriched.drop(columns=['visual_embedding'])

    # FIX: Also drop it from the original df so we can use it for baseline prediction
    if 'visual_embedding' in df.columns:
        df = df.drop(columns=['visual_embedding'])

    train_df_original, inference_df_original, test_df_original = split_ts_dataset(df)
    pred_df_original = predict(pipeline, train_df_original, inference_df_original, test_df_original, CONTEXT_LENGTH, PREDICTION_LENGTH, COVARIATE_COLUMNS)

    # Use + for list concatenation to create a new list, or use the extended name if modified in place
    # Extending COVARIATE_COLUMNS in place (if it's a list)
    final_cov = COVARIATE_COLUMNS + cov_cols

    train_df, inference_df, test_df = split_ts_dataset(df_enriched)

    # 4. Predict
    pred_df_multimodal = predict(pipeline, train_df, inference_df, test_df, CONTEXT_LENGTH, PREDICTION_LENGTH, final_cov)

    models_to_plot = {
        #"Zero-Shot (Base)": pred_df_original,
        "Multimodal": pred_df_multimodal,
    }
    # 5. Visualize
    # Pass df as both train and test since we just sliced from it
    plot_model_comparison(
        train_df=train_df,
        test_df=test_df,
        model_predictions=models_to_plot,
        plot_history_length=200,   # How many historical steps to show before the forecast
        prediction_length=PREDICTION_LENGTH,
        seasonality=96             # Seasonality for MASE (96 steps = 48 hours for 30min data)
    )

if __name__ == "__main__":
    main()
