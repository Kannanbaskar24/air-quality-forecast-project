import pandas as pd

def check_drift(new_data: pd.Series, old_data: pd.Series, threshold=0.2):
    """
    Returns True if mean difference > threshold fraction of old mean.
    """
    if old_data.empty or new_data.empty:
        return False
    mean_old = old_data.mean()
    mean_new = new_data.mean()
    diff = abs(mean_new - mean_old) / (mean_old + 1e-6)
    return diff > threshold
