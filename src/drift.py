import pandas as pd

def check_drift(new_data: pd.Series, old_data: pd.Series, threshold=0.2, min_samples=10):
    """
    Returns True if mean difference > threshold fraction of old mean.
    Requires at least min_samples data points in both series.
    """
    new_data_clean = new_data.dropna()
    old_data_clean = old_data.dropna()
    if len(old_data_clean) < min_samples or len(new_data_clean) < min_samples:
        return False
    
    mean_old = old_data_clean.mean()
    mean_new = new_data_clean.mean()
    diff = abs(mean_new - mean_old) / (mean_old + 1e-6)
    return diff > threshold
