import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_item_mase(y_true, y_pred, y_history):
    """
    Calculates MASE for a single item.
    MASE = MAE(forecast) / MAE(naive_history)
    """
    #Calculate MAE of the forecast 
    mae_forecast = np.mean(np.abs(y_true.values - y_pred.values))

    # Calculate MAE 
    if len(y_history) < 2:
        return np.nan # Not enough history

    naive_errors = np.abs(np.diff(y_history.values))
    mae_naive = np.mean(naive_errors)

    if mae_naive == 0:
        return np.inf if mae_forecast > 0 else 0.0

    return mae_forecast / mae_naive

def calculate_item_mape(y_true, y_pred, epsilon=1e-10):
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    """
    # Using numpy values to ensure compatibility
    y_true_vals = y_true.values if hasattr(y_true, 'values') else y_true
    y_pred_vals = y_pred.values if hasattr(y_pred, 'values') else y_pred
    
    mask = y_true_vals > epsilon
    if np.sum(mask) == 0:
        return np.nan 

    mape = np.mean(np.abs((y_true_vals[mask] - y_pred_vals[mask]) / y_true_vals[mask])) * 100
    return mape

def calculate_item_wmape(y_true, y_pred):
    """
    Calculates Weighted Mean Absolute Percentage Error (wMAPE).
    """
    y_true_vals = y_true.values if hasattr(y_true, 'values') else y_true
    y_pred_vals = y_pred.values if hasattr(y_pred, 'values') else y_pred
    
    total_abs_error = np.sum(np.abs(y_true_vals - y_pred_vals))
    total_actuals = np.sum(np.abs(y_true_vals))

    if total_actuals == 0:
        return np.inf

    return (total_abs_error / total_actuals) * 100

def plot_prediction(past_data, full_data, bolt_predictor, c2_predictions, known_covariates_future, save_dir=None):
    """
    past_data: The data used for input (truncated)
    full_data: The original full data (containing the ground truth for evaluation)
    """
    print("\nGenerating Backtest Forecasts (Hiding last steps)...")
    PLOT_LENGTH= 200
    PREDICTION_LENGTH = 96 
    
    # Generate Bolt predictions
    model_names = bolt_predictor.model_names()
    model_predictions = {}

    for model_name in model_names:
        print(f"Predicting with {model_name}...")
        model_predictions[model_name] = bolt_predictor.predict(past_data, known_covariates=known_covariates_future, model=model_name)

    for c2_name, c2_pred in c2_predictions.items():
        # Append to the list of names so the plotter loops over it
        model_names.append(c2_name)
        # Store the prediction dataframe
        model_predictions[c2_name] = c2_pred

    # Setup Plotting
    item_ids = sorted(full_data.reset_index()['item_id'].unique())
    num_plots = len(item_ids)
    
    # Safety check for huge datasets
    if num_plots > 20:
        print(f"Warning: Plotting {num_plots} series will create a very large image.")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    fig, axs = plt.subplots(num_plots, 1, figsize=(15, 6 * num_plots), sharex=False)
    if num_plots == 1: axs = [axs]

    metrics_summary = {"MASE": [], "MAPE": [], "wMAPE": []}

    for i, item_id in enumerate(item_ids):
        ax = axs[i]

        # Prepare Data 
        history_context = full_data.loc[item_id].iloc[-(PLOT_LENGTH + PREDICTION_LENGTH):]

        # Ground Truth
        ground_truth_future = full_data.loc[item_id].iloc[-PREDICTION_LENGTH:]['pv_value']

        history_for_metric = full_data.loc[item_id].iloc[:-PREDICTION_LENGTH]['pv_value']

        # Plot Ground Truth
        ax.plot(history_context.index, history_context['pv_value'],
                label='Actual Ground Truth', color='black', linewidth=2, alpha=0.6)

        # Highlight the Forecast Window
        cutoff_date = ground_truth_future.index[0]
        ax.axvspan(cutoff_date, ground_truth_future.index[-1], color='gray', alpha=0.1, label="Forecast Window")

        # Plot Each Model 
        for idx, model_name in enumerate(model_names):
            preds = model_predictions[model_name]
            forecast = preds.loc[item_id]

            # Calculate Metrics
            try:
                mase_score = calculate_item_mase(
                    y_true=ground_truth_future,
                    y_pred=forecast['mean'],
                    y_history=history_for_metric
                )
                mape_score = calculate_item_mape(ground_truth_future, forecast['mean'])
                wmape_score = calculate_item_wmape(ground_truth_future, forecast['mean'])
                
                metrics_label = f"MASE:{mase_score:.2f} MAPE:{mape_score:.0f}% wMAPE:{wmape_score:.0f}%"
            except Exception as e:
                print(f"Error calc metrics for {model_name}: {e}")
                metrics_label = "Metrics: N/A"

            # Prepare Label
            clean_name = model_name.split('/')[-1]
            label_text = f'{clean_name} | {metrics_label}'
            color = colors[idx % len(colors)]

            # Plot Mean
            ax.plot(forecast.index, forecast['mean'],
                    label=label_text, color=color, linewidth=2, linestyle='--')

            # Plot Confidence Interval
            if '0.1' in forecast.columns and '0.9' in forecast.columns:
                ax.fill_between(
                    forecast.index,
                    forecast['0.1'],
                    forecast['0.9'],
                    color=color, alpha=0.15
                )

        # Formatting
        ax.set_title(f"Segment {item_id}: Backtest Performance", fontsize=14, fontweight='bold')
        ax.set_ylabel("PV Value", fontsize=10)
        ax.axvline(x=cutoff_date, color='red', linestyle=':', linewidth=1.5, label="Prediction Start")
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "baseline_backtest.png")
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        
    plt.show()
    plt.close()
