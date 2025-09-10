from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
import tensorflow as tf
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

app = FastAPI(title="Stock Prediction API")

# ----------- Input Schema -----------
class StockRequest(BaseModel):
    ticker: str
    future_days: int = 30
    start_date: str = "2021-01-01"
    end_date: str = "2024-10-31"


# ----------- Helper Functions -----------
def create_model(time_steps, n_features):
    model = Sequential()
    model.add(Input(shape=(time_steps, n_features)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# ----------- API Endpoint -----------
@app.post("/predict")
def predict_stock(request: StockRequest):
    try:
        # 1. Download Data
        data = yf.download(request.ticker, start=request.start_date, end=request.end_date, interval="1d")
        if data.empty:
            return {"error": "No stock data found for given ticker & dates."}

        # 2. Prepare Features
        feature_columns = ["Close"]
        features = data[feature_columns].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(features)

        time_steps = 60
        X, y = [], []
        for i in range(len(scaled_data) - time_steps):
            X.append(scaled_data[i:i + time_steps])
            y.append(scaled_data[i + time_steps, 0])
        X, y = np.array(X), np.array(y)

        # 3. Train Model (small epochs for demo)
        model = create_model(time_steps, X.shape[2])
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

        # 4. Predict Future
        last_sequence = X[-1]
        future_predictions = []
        seq = last_sequence.copy()

        for _ in range(request.future_days):
            pred = model.predict(seq.reshape(1, *seq.shape), verbose=0)
            dummy = np.zeros((1, scaler.scale_.shape[0]))
            dummy[0, 0] = pred.item()
            price = scaler.inverse_transform(dummy)[0, 0]
            future_predictions.append(price)
            seq = np.roll(seq, -1, axis=0)
            seq[-1, 0] = pred.item()

        future_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=request.future_days, freq="B")

        # 5. Only Predictions Table (no chart)
        predictions_table = [
            {"date": str(d.date()), "predicted_price": round(float(p), 2)}
            for d, p in zip(future_dates, future_predictions)
        ]

        return {"predictions": predictions_table}

    except Exception as e:
        return {"error": str(e)}
