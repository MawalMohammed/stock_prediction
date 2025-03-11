# app.py
from flask import Flask, render_template, request
from model_prediction import load_saved_model, predict_next_n_days

app = Flask(__name__)

# Replace with your API key
api_key = '8XM5MUWUN6HT1W7H'

# Load the model and scaler
model, scaler = load_saved_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']
        prediction = predict_next_n_days(stock_symbol, model, scaler, api_key)
        return render_template('result.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
