# model_prediction.py
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from data_preprocessing import fetch_data, preprocess_data

def load_saved_model(model_path='my_model.keras', scaler_path='scaler.save'):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def preprocess_latest_data(data, scaler, seq_length=30):
    data = data[['open', 'high', 'low', 'close', 'volume']]
    data_scaled = scaler.transform(data)
    last_sequence = data_scaled[-seq_length:]
    return last_sequence

def predict_next_four_days(stock_symbol, model, scaler, seq_length=30):
    latest_data = fetch_data(api_key, stock_symbol)
    last_sequence = preprocess_latest_data(latest_data, scaler, seq_length)
    predictions = []

    for _ in range(4):
        input_sequence = last_sequence.reshape(1, seq_length, 5)
        next_day_prediction = model.predict(input_sequence)
        predictions.append(next_day_prediction[0][0])
        last_sequence = np.append(last_sequence[1:], np.hstack([last_sequence[-1, :4], next_day_prediction[0]]).reshape(1, -1), axis=0)

    predictions_rescaled = scaler.inverse_transform(np.hstack([np.zeros((len(predictions), 3)), np.array(predictions).reshape(-1, 1), np.zeros((len(predictions), 1))]))[:, 3]
    return predictions_rescaled.flatten()