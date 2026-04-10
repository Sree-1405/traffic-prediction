import numpy as np

def create_labels(data):
    """
    Converts speed values into traffic classes:
    0 → Low congestion
    1 → Moderate congestion
    2 → High congestion
    """

    data = np.array(data)

    # -----------------------------
    # Handle different dimensions
    # -----------------------------
    if data.ndim == 1:
        mean_speed = data
    elif data.ndim == 2:
        mean_speed = np.mean(data, axis=1)
    elif data.ndim == 3:
        mean_speed = np.mean(data, axis=(1, 2))
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")

    # -----------------------------
    # Speed-based labeling (METR-LA)
    # -----------------------------
    labels = np.zeros_like(mean_speed, dtype=int)

    labels[mean_speed < 30] = 2        # High congestion
    labels[(mean_speed >= 30) & (mean_speed < 60)] = 1  # Moderate
    labels[mean_speed >= 60] = 0       # Low congestion

    return labels
