# model_training.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import numpy as np

def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, x_train, y_train, epochs=100, batch_size=32, validation_split=0.1):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping])
    return model, history

def evaluate_model(model, x_test, y_test, scaler, test_data_scaled):  # Add test_data_scaled as argument
    y_pred = model.predict(x_test)
    y_pred_rescaled = scaler.inverse_transform(np.hstack([np.zeros((len(y_pred), 3)), y_pred.reshape(-1, 1), np.zeros((len(y_pred), 1))]))[:, 3]
    y_test_rescaled = scaler.inverse_transform(test_data_scaled)[30:, 3]  # Use test_data_scaled
    
    valid_indices = ~np.isnan(y_test_rescaled) & ~np.isnan(y_pred_rescaled)
    y_test_rescaled_clean = y_test_rescaled[valid_indices]
    y_pred_rescaled_clean = y_pred_rescaled[valid_indices]
    
    mse = mean_squared_error(y_test_rescaled_clean, y_pred_rescaled_clean)
    mae = mean_absolute_error(y_test_rescaled_clean, y_pred_rescaled_clean)
    
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

def save_model(model, scaler, model_path='my_model.keras', scaler_path='scaler.save'):
    model.save(model_path)
    joblib.dump(scaler, scaler_path)