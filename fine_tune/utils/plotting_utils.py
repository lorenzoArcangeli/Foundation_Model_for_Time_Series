import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def calculate_item_mase(y_true, y_pred, y_history, seasonality=96):
    """Calculates MASE for a single item."""
    # Mean Absolute Error of the forecast
    forecast_mae = np.mean(np.abs(y_true - y_pred))

    # Mean Absolute Error of the naive baseline on history
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
    """
    # Filter out zeros (standard for PV data)
    # We only compute MAPE when actual production > 0
    mask = y_true > epsilon
    if np.sum(mask) == 0:
        return np.nan # No valid data points (night time)

    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

def calculate_item_wmape(y_true, y_pred):
    """
    Calculates Weighted Mean Absolute Percentage Error (wMAPE).
    """
    total_abs_error = np.sum(np.abs(y_true - y_pred))
    total_actuals = np.sum(np.abs(y_true))

    if total_actuals == 0:
        return np.inf

    return (total_abs_error / total_actuals) * 100

def plot_model_comparison(train_df, test_df, model_predictions,
                          plot_history_length=200, prediction_length=96, seasonality=96,
                          save_dir=None):
    """
    Plots model comparison and saves to save_dir if provided.
    """
    
    # Setup Data
    # Concatenate train and test to get the full timeline for plotting context
    full_data = pd.concat([train_df, test_df]).sort_values(['item_id', 'timestamp'])

    item_ids = sorted(full_data['item_id'].unique())
    num_plots = len(item_ids)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axs = plt.subplots(num_plots, 1, figsize=(15, 6 * num_plots), sharex=False)
    if num_plots == 1: axs = [axs]

    print(f"Generating plots for {num_plots} items...")

    for i, item_id in enumerate(item_ids):
        ax = axs[i]

        # Filter data for this specific item
        item_full_data = full_data[full_data['item_id'] == item_id].set_index('timestamp')
        item_test_data = test_df[test_df['item_id'] == item_id].set_index('timestamp')
        item_train_data = train_df[train_df['item_id'] == item_id].set_index('timestamp')

        # Last 'plot_history_length' points from training + the prediction window
        history_context = item_full_data.iloc[-(plot_history_length + prediction_length):]

        ground_truth_future = item_test_data['pv_value']

        history_for_metric = item_train_data['pv_value'].values

        # Plot
        ax.plot(history_context.index, history_context['pv_value'],
                label='Actual Ground Truth', color='black', linewidth=2, alpha=0.6)

        # Forecast Window
        if len(ground_truth_future) > 0:
            cutoff_date = ground_truth_future.index[0]
            ax.axvspan(cutoff_date, ground_truth_future.index[-1], color='gray', alpha=0.1, label="Forecast Window")
            ax.axvline(x=cutoff_date, color='black', linestyle=':', linewidth=1)

        # Plot Each Model
        for idx, (model_name, pred_df_all) in enumerate(model_predictions.items()):
            item_preds = pred_df_all[pred_df_all['item_id'] == item_id].set_index('timestamp')

            if item_preds.empty:
                continue

            # Calculate MASE
            try:
                y_pred = item_preds['predictions'].values[-len(ground_truth_future):]
                y_true = ground_truth_future.values[-len(y_pred):]

                mase_score = calculate_item_mase(
                    y_true=y_true,
                    y_pred=y_pred,
                    y_history=history_for_metric,
                    seasonality=seasonality
                )
                # MAPE and wMAPE
                mape_score = calculate_item_mape(y_true, y_pred)
                wmape = calculate_item_wmape(y_true, y_pred)

                metrics_label = f"MASE: {mase_score:.2f} | MAPE: {mape_score:.1f}% | wMAPE: {wmape: .1f}"
            except Exception as e:
                metrics_label = "MASE: N/A"

            color = colors[idx % len(colors)]
            label_text = f'{model_name} | {metrics_label}'

            ax.plot(item_preds.index, item_preds['predictions'],
                    label=label_text, color=color, linewidth=2, linestyle='--')

            if '0.1' in item_preds.columns and '0.9' in item_preds.columns:
                ax.fill_between(
                    item_preds.index,
                    item_preds['0.1'],
                    item_preds['0.9'],
                    color=color, alpha=0.15
                )

        # Final formatting
        ax.set_title(f"Item {item_id}: Forecast Comparison", fontsize=14, fontweight='bold')
        ax.set_ylabel("PV Value")
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "model_comparison.png")
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()
    plt.close()
