import numpy as np

def arima_predict(train, test):
    """
    Simple persistence ARIMA-style baseline
    """
    last_value = np.mean(train, axis=0)
    preds = np.tile(last_value, (len(test), 1))
    return preds
