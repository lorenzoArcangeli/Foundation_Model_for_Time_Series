import numpy as np

def calculate_item_mase(y_true, y_pred, y_history, seasonality=96):
    """Calculates MASE for a single item."""
    forecast_mae = np.mean(np.abs(y_true - y_pred))
    if len(y_history) <= seasonality:
        return np.inf 
    naive_errors = np.abs(y_history[seasonality:] - y_history[:-seasonality])
    naive_mae = np.mean(naive_errors)
    if naive_mae == 0:
        return np.inf
    return forecast_mae / naive_mae

def calculate_item_mape(y_true, y_pred, epsilon=1e-10):
    mask = y_true > epsilon
    if np.sum(mask) == 0:
        return np.nan 
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

def calculate_item_wmape(y_true, y_pred):
    total_abs_error = np.sum(np.abs(y_true - y_pred))
    total_actuals = np.sum(np.abs(y_true))
    if total_actuals == 0:
        return np.inf
    return (total_abs_error / total_actuals) * 100

def calculate_item_wql(y_true, quantile_preds, quantiles):
    total_loss = 0
    total_abs_target = np.sum(np.abs(y_true))
    if total_abs_target == 0:
        return np.inf
    for q in quantiles:
        y_pred_q = quantile_preds[q]
        errors = y_true - y_pred_q
        loss = np.maximum(q * errors, (q - 1) * errors)
        total_loss += np.sum(2 * loss)
    wql = total_loss / (len(quantiles) * total_abs_target)
    return wql

def calculate_item_sql(y_true, quantile_preds, quantiles, y_history, seasonality=96):
    total_loss = 0
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
        total_loss += np.mean(2 * loss)
    sql = total_loss / (len(quantiles) * naive_mae)
    return sql
