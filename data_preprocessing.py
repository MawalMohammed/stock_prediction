# data_preprocessing.py
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def fetch_data(api_key, stock_symbol):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=stock_symbol, outputsize='full')
    data = data.reset_index()
    data = data.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'})
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    data = data.dropna()
    return data

def preprocess_data(data, test_size=0.2):
    # Split data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
    
    # Scale the data
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data[['open', 'high', 'low', 'close', 'volume']])
    test_data_scaled = scaler.transform(test_data[['open', 'high', 'low', 'close', 'volume']])
    
    return train_data_scaled, test_data_scaled, scaler

def create_sequences(data, seq_length=30):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 3])  # 'close' is the 4th column (index 3)
    return np.array(x), np.array(y)