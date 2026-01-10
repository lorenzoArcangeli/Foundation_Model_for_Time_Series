import numpy as np
import pandas as pd
from plotting_utils import calculate_item_mase, calculate_item_mape, calculate_item_wmape

def backtest_model(pipeline, df, num_windows=3, step_size=96, prediction_length=96):
    """
    Performs Rolling Origin Cross-Validation (Backtesting).
    """
    print(f"\n--- Starting Robust Backtest ({num_windows} windows) ---")

    metrics = {"mase": [], "mape": [], "wmape": []}

    for window_idx in range(num_windows):
        cutoff_step = window_idx * step_size
        
        test_df_list = []
        context_df_list = []

        grouped = df.groupby('item_id')

        for item_id, group in grouped:
            total_len = len(group)
            end_idx = total_len - cutoff_step
            split_idx = end_idx - prediction_length

            if split_idx <= 0: continue 

            context_data = group.iloc[:split_idx]
            ground_truth = group.iloc[split_idx:end_idx]

            context_df_list.append(context_data)
            test_df_list.append(ground_truth)

        if not context_df_list:
            continue
            
        window_context_df = pd.concat(context_df_list)
        window_test_df = pd.concat(test_df_list)

        print(f"Window {window_idx+1}/{num_windows}: Predicting...")

        forecast = pipeline.predict_df(
            df=window_context_df,
            prediction_length=prediction_length,
            id_column="item_id",
            timestamp_column="timestamp",
            target="pv_value"
        )

        w_mase, w_mape, w_wmape = [], [], []

        for item_id in window_test_df['item_id'].unique():
            y_pred = forecast[forecast['item_id'] == item_id]['predictions'].values
            y_true = window_test_df[window_test_df['item_id'] == item_id]['pv_value'].values
            y_history = window_context_df[window_context_df['item_id'] == item_id]['pv_value'].values

            if len(y_pred) != len(y_true): continue

            mase = calculate_item_mase(y_true, y_pred, y_history)
            mape = calculate_item_mape(y_true, y_pred)
            wmape = calculate_item_wmape(y_true, y_pred)

            if not np.isinf(mase): w_mase.append(mase)
            if not np.isnan(mape): w_mape.append(mape)
            if not np.isnan(wmape): w_wmape.append(wmape)

        if w_mase:
            avg_mase = np.mean(w_mase)
            metrics["mase"].append(avg_mase)
        else:
            avg_mase = np.nan
            
        if w_mape:
            avg_mape = np.mean(w_mape)
            metrics["mape"].append(avg_mape)
        else:
            avg_mape = np.nan
            
        if w_wmape:
            avg_wmape = np.mean(w_wmape)
            metrics["wmape"].append(avg_wmape)

        print(f"Window {window_idx+1} Metrics -> MASE: {avg_mase:.3f}, MAPE: {avg_mape:.1f}%, WMAPE: {avg_wmape:.1f}%")

    print(f"\n--- Robust Result (Average of {num_windows} windows) ---")
    print(f"Robust MASE: {np.nanmean(metrics['mase']):.3f}")
    print(f"Robust MAPE: {np.nanmean(metrics['mape']):.1f}%")
    print(f"Robust WMAPE: {np.nanmean(metrics['wmape']):.1f}%")

    return metrics