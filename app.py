# app.py
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib
from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import os

# --- Configuration ---
TIME_STEPS = 60      # Must be the same as the value used for training
FUTURE_DAYS = 30     # Number of days to predict into the future
HISTORICAL_DAYS = 200 # How many past days of data to return for the chart
MODEL_DIR = "models" # Directory where your .h5 and .pkl files are stored

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Prediction Helper Functions ---
def predict_future(model, last_sequence, scaler, n_future):
    """Predicts future stock values using a pre-trained model."""
    future_predictions_scaled = []
    current_sequence = last_sequence.copy().reshape(1, TIME_STEPS, 1)

    for _ in range(n_future):
        next_pred_scaled = model.predict(current_sequence, verbose=0)
        future_predictions_scaled.append(next_pred_scaled[0, 0])
        current_sequence = np.append(current_sequence[:, 1:, :], [[next_pred_scaled]], axis=1)

    future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    return future_predictions

def generate_future_dates(last_date, num_days):
    """Generates future business day dates."""
    return pd.bdate_range(start=last_date + timedelta(days=1), periods=num_days)

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def get_prediction():
    # --- 1. Get Ticker from POST Request Body ---
    json_data = request.get_json()
    if not json_data or 'ticker' not in json_data:
        return jsonify({"error": "Invalid input. Please provide a JSON body with a 'ticker' key."}), 400
    
    ticker = json_data['ticker']
    print(f"Received prediction request for: {ticker}")

    # --- 2. Check if Model Files Exist ---
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return jsonify({"error": f"Model for ticker '{ticker}' not found. Please train the model first."}), 404

    try:
        # --- 3. Load the Pre-trained Model and Scaler ---
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        # --- 4. Fetch Recent Historical Data for Charting and Prediction ---
        end_date = datetime.today()
        start_date = end_date - timedelta(days=HISTORICAL_DAYS + TIME_STEPS) # Fetch enough data
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return jsonify({"error": f"Could not fetch data for ticker '{ticker}'."}), 500

        # --- 5. Prepare the Last Sequence for Prediction ---
        last_sequence_raw = data['Close'].values[-TIME_STEPS:].reshape(-1, 1)
        last_sequence_scaled = scaler.transform(last_sequence_raw)
        
        # --- 6. Make Predictions ---
        predictions = predict_future(model, last_sequence_scaled, scaler, FUTURE_DAYS)
        
        # --- 7. Generate Future Dates ---
        last_date = data.index[-1]
        future_dates = generate_future_dates(last_date, FUTURE_DAYS)
        
        # --- 8. Format the Response for FlutterFlow ---
        # Get the historical data for the chart
        historical_chart_data = data.tail(HISTORICAL_DAYS)

        chart_data = []
        # Add historical points
        for index, row in historical_chart_data.iterrows():
            chart_data.append({
                "date": index.strftime('%Y-%m-%d'),
                "price": row['Close'],
                "type": "historical"
            })
        
        # Add predicted points
        for date, price in zip(future_dates, predictions):
            chart_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "price": float(price[0]),
                "type": "predicted"
            })
            
        response = {
            "ticker": ticker,
            "chartData": chart_data
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    # On Render, the PORT environment variable is used. Default to 5000 for local testing.
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)