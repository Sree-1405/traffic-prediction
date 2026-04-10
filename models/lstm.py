from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

def train_lstm(train_data):
    model = Sequential([
        Input(shape=(train_data.shape[1], 1)),
        LSTM(16),                 # 🔻 very small
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    # 🔻 VERY FEW epochs
    model.fit(
        train_data.reshape(train_data.shape[0], train_data.shape[1], 1),
        train_data[:, -1],
        epochs=3,                # 🔻 intentionally low
        batch_size=64,
        verbose=0
    )
    return model


def predict_lstm(model, test_data):
    return model.predict(
        test_data.reshape(test_data.shape[0], test_data.shape[1], 1),
        verbose=0
    )
