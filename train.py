# train.py
from data_preprocessing import fetch_data, preprocess_data, create_sequences
from model_training import build_model, train_model, evaluate_model, save_model

# Replace with your API key
api_key = '8XM5MUWUN6HT1W7H'

# Step 1: Fetch and preprocess the data
stock_symbol = 'AAPL'  # Replace with the desired stock symbol
data = fetch_data(api_key, stock_symbol)
train_data_scaled, test_data_scaled, scaler = preprocess_data(data)

# Step 2: Create sequences for training and testing
x_train, y_train = create_sequences(train_data_scaled)
x_test, y_test = create_sequences(test_data_scaled)

# Step 3: Build the model
model = build_model((x_train.shape[1], x_train.shape[2]))

# Step 4: Train the model
model, history = train_model(model, x_train, y_train)

# Step 5: Evaluate the model
evaluate_model(model, x_test, y_test, scaler, test_data_scaled)  # Pass test_data_scaled

# Step 6: Save the model and scaler
save_model(model, scaler)