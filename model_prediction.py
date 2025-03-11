# model_prediction.py
import joblib
from tensorflow.keras.models import load_model
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Replace with your API key
api_key = '8XM5MUWUN6HT1W7H'

# Initialize the TimeSeries object
ts = TimeSeries(key=api_key, output_format='pandas')

# Function to load the saved model and scaler
def load_saved_model(model_path='my_model.keras', scaler_path='scaler.save'):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Function to fetch the latest stock data
def fetch_latest_data(stock_symbol):
    data, meta_data = ts.get_daily(symbol=stock_symbol, outputsize='compact')
    data = data.reset_index()
    data = data.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'})
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    return data

# Function to preprocess the latest data
def preprocess_latest_data(data, scaler, seq_length=30):
    # Ensure the DataFrame has the correct columns in the correct order
    data = data[['open', 'high', 'low', 'close', 'volume']]
    
    # Debugging: Print the data being passed to scaler.transform()
    print("Data being scaled:")
    print(data.head())
    
    # Scale the data
    data_scaled = scaler.transform(data)
    
    # Use the last `seq_length` days as the input sequence
    last_sequence = data_scaled[-seq_length:]
    return last_sequence

# Function to predict the next `n` days
def predict_next_n_days(stock_symbol, model, scaler, api_key, seq_length=30, n=4):
    # Fetch the latest data
    latest_data = fetch_latest_data(stock_symbol)
    
    # Preprocess the latest data
    last_sequence = preprocess_latest_data(latest_data, scaler, seq_length)
    predictions = []

    for _ in range(n):  # Predict the next `n` days
        # Reshape the sequence to match the model's input shape
        input_sequence = last_sequence.reshape(1, seq_length, 5)
        
        # Make a prediction
        next_day_prediction = model.predict(input_sequence)
        
        # Append the prediction to the results
        predictions.append(next_day_prediction[0][0])
        
        # Update the sequence by removing the oldest data point and adding the new prediction
        last_sequence = np.append(last_sequence[1:], np.hstack([last_sequence[-1, :4], next_day_prediction[0]]).reshape(1, -1), axis=0)

    # Inverse scale the predictions to get actual stock prices
    predictions_rescaled = scaler.inverse_transform(np.hstack([np.zeros((len(predictions), 3)), np.array(predictions).reshape(-1, 1), np.zeros((len(predictions), 1))]))[:, 3]
    
    return predictions_rescaled.flatten()

