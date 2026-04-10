import h5py
import numpy as np

def load_metr_la(path):
    """
    Loads METR-LA speed data
    Output shape: (time_steps, num_sensors)
    """
    with h5py.File(path, "r") as f:
        data = np.array(f["df"]["block0_values"])
    return data


def train_test_split(data, split=0.7):
    n = int(len(data) * split)
    return data[:n], data[n:]








"""import h5py
import numpy as np

def load_metr_la(path):
    with h5py.File(path, "r") as f:
        data = f["df"][:]
    return data

def train_test_split(data, split=0.8):
    T = data.shape[0]
    idx = int(T * split)
    return data[:idx], data[idx:]
"""