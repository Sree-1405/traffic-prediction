from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input

def train_gru(train_data):
    model = Sequential([
        Input(shape=(train_data.shape[1], 1)),
        GRU(16),                  # 🔻 small GRU
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    model.fit(
        train_data.reshape(train_data.shape[0], train_data.shape[1], 1),
        train_data[:, -1],
        epochs=3,                # 🔻 few epochs
        batch_size=64,
        verbose=0
    )
    return model


def predict_gru(model, test_data):
    return model.predict(
        test_data.reshape(test_data.shape[0], test_data.shape[1], 1),
        verbose=0
    )
