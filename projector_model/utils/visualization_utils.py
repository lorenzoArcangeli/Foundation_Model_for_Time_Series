import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from . import metrics
from . import config

def log_validation_metrics(train_df, test_df, model_predictions, item_id, phase_info, output_dir, 
                          prediction_length=config.PREDICTION_LENGTH, seasonality=config.SEASONALITY):
    
    # Setup Data
    full_data = pd.concat([train_df, test_df]).sort_values(['item_id', 'timestamp'])
    
    # Filter data for this specific item
    item_full_data = full_data[full_data['item_id'] == item_id].set_index('timestamp')
    item_test_data = test_df[test_df['item_id'] == item_id].set_index('timestamp')
    item_train_data = train_df[train_df['item_id'] == item_id].set_index('timestamp')

    ground_truth_future = item_test_data['pv_value']
    history_for_metric = item_train_data['pv_value'].values

    log_line = ""
    
    for idx, (model_name, pred_df_all) in enumerate(model_predictions.items()):
        item_preds = pred_df_all[pred_df_all['item_id'] == item_id].set_index('timestamp')
        
        if item_preds.empty:
            continue

        # Metrics
        try:
            y_pred_median = item_preds['predictions'].values[-len(ground_truth_future):]
            y_true = ground_truth_future.values[-len(y_pred_median):]

            mase_score = metrics.calculate_item_mase(y_true, y_pred_median, history_for_metric, seasonality)
            mape_score = metrics.calculate_item_mape(y_true, y_pred_median)
            wmape = metrics.calculate_item_wmape(y_true, y_pred_median)
            
            # WQL & SQL
            quantiles_to_check = [0.1, 0.5, 0.9]
            quantile_preds_dict = {}
            has_quantiles = True
            
            for q in quantiles_to_check:
                q_str = str(q)
                if q_str in item_preds.columns:
                    quantile_preds_dict[q] = item_preds[q_str].values[-len(ground_truth_future):]
                else:
                    has_quantiles = False
                    if q == 0.5: quantile_preds_dict[0.5] = y_pred_median
            
            if has_quantiles:
                wql_score = metrics.calculate_item_wql(y_true, quantile_preds_dict, quantiles_to_check)
                sql_score = metrics.calculate_item_sql(y_true, quantile_preds_dict, quantiles_to_check, history_for_metric, seasonality)
                metrics_str = (f"MASE: {mase_score:.2f} MAPE: {mape_score:.0f}% wMAPE: {wmape:.1f}% "
                                 f"WQL: {wql_score:.3f} SQL: {sql_score:.3f}")
            else:
                metrics_str = f"MASE: {mase_score:.2f} MAPE: {mape_score:.0f}% wMAPE: {wmape:.1f}%"

        except Exception as e:
            metrics_str = f"Error: {e}"

        item_log = f"- Item {item_id}: {metrics_str}"
        log_line += item_log + "\n"

    return log_line

def plot_model_comparison(train_df, test_df, model_predictions,
                          plot_history_length=200, prediction_length=config.PREDICTION_LENGTH, seasonality=config.SEASONALITY, output_dir="."):

    # Setup Data
    # Concatenate train and test to get the full timeline for plotting context
    full_data = pd.concat([train_df, test_df]).sort_values(['item_id', 'timestamp'])

    item_ids = sorted(full_data['item_id'].unique())
    num_plots = len(item_ids)

    # Safety check for very large datasets
    if num_plots > 20:
        print(f"Warning: Plotting {num_plots} series will create a very large image.")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Blue, Orange, Green, Red

    # Initialize Figure
    fig, axs = plt.subplots(num_plots, 1, figsize=(15, 6 * num_plots), sharex=False)
    if num_plots == 1: axs = [axs] # Ensure axs is always iterable

    print(f"Generating plots for {num_plots} items...")

    for i, item_id in enumerate(item_ids):
        ax = axs[i]

        # Prepare Series Data
        # Filter data for this specific item
        item_full_data = full_data[full_data['item_id'] == item_id].set_index('timestamp')
        item_test_data = test_df[test_df['item_id'] == item_id].set_index('timestamp')
        item_train_data = train_df[train_df['item_id'] == item_id].set_index('timestamp')

        # History Context
        # We grab the last 'plot_history_length' points from training + the prediction window
        history_context = item_full_data.iloc[-(plot_history_length + prediction_length):]

        # Ground Truth (The actual future)
        ground_truth_future = item_test_data['pv_value']

        # History for Metric (for MASE denominator)
        history_for_metric = item_train_data['pv_value'].values

        # Plot Background
        # Plot the actual values (history + future)
        ax.plot(history_context.index, history_context['pv_value'],
                label='Actual Ground Truth', color='black', linewidth=2, alpha=0.6)

        # Highlight the Forecast Window
        if len(ground_truth_future) > 0:
            cutoff_date = ground_truth_future.index[0]
            ax.axvspan(cutoff_date, ground_truth_future.index[-1], color='gray', alpha=0.1, label="Forecast Window")
            ax.axvline(x=cutoff_date, color='black', linestyle=':', linewidth=1)

        # Plot Each Model
        for idx, (model_name, pred_df_all) in enumerate(model_predictions.items()):
            # Filter predictions for this item
            item_preds = pred_df_all[pred_df_all['item_id'] == item_id].set_index('timestamp')

            if item_preds.empty:
                continue

            # Calculate Metrics
            try:
                # Align lengths (take last N points if necessary)
                y_pred_median = item_preds['predictions'].values[-len(ground_truth_future):]
                y_true = ground_truth_future.values[-len(y_pred_median):]

                # MASE
                mase_score = metrics.calculate_item_mase(
                    y_true=y_true,
                    y_pred=y_pred_median,
                    y_history=history_for_metric,
                    seasonality=seasonality
                )
                # MAPE
                mape_score = metrics.calculate_item_mape(y_true, y_pred_median)
                # wMAPE
                wmape = metrics.calculate_item_wmape(y_true, y_pred_median)

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
                    wql_score = metrics.calculate_item_wql(y_true, quantile_preds_dict, quantiles_to_check)
                    sql_score = metrics.calculate_item_sql(y_true, quantile_preds_dict, quantiles_to_check, history_for_metric, seasonality)

                    metrics_label = (f"MASE:{mase_score:.2f} MAPE:{mape_score:.0f}% wMAPE:{wmape:.1f}% "
                                     f"WQL:{wql_score:.3f} SQL:{sql_score:.3f}")
                else:
                    metrics_label = f"MASE:{mase_score:.2f} MAPE:{mape_score:.0f}% wMAPE:{wmape:.1f}%"

            except Exception as e:
                print(f"Error calculating metrics for {model_name} on item {item_id}: {e}")
                metrics_label = "Metrics: Error"

            # Plot Predictions
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

        # Final Formatting
        ax.set_title(f"Item {item_id}: Forecast Comparison", fontsize=14, fontweight='bold')
        ax.set_ylabel("PV Value")
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    
    #If figure already present save with an incremental id
    base_name = f"forecast_comparison_{item_id}"
    save_path = os.path.join(output_dir, f"{base_name}.png")
    
    if os.path.exists(save_path):
        i=1
        while os.path.exists(os.path.join(output_dir, f"{base_name}_{i}.png")):
             i+=1
        save_path = os.path.join(output_dir, f"{base_name}_{i}.png")
        
    plt.savefig(save_path)

    plt.tight_layout()
    plt.show()
