from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from datetime import datetime, timedelta
import requests

app = Flask(__name__, template_folder="templates")

# LSTM Model Creation
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    lr_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.95, staircase=True)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mean_squared_error')
    return model

# Fetch Stock Data
def fetch_stock_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    return yf.download(ticker, start=start_date, end=end_date)

# Fetch Sentiment Data
def fetch_sentiment_data(ticker):
    api_key = "993f2e8f447d45a5bc302486bdb255a2"  
    sentiment_start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    url = f"https://newsapi.org/v2/everything?q={ticker}&from={sentiment_start_date}&sortBy=publishedAt&apiKey={api_key}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"API Error for {ticker}: {response.json().get('message', 'Unknown error')}")
            return pd.Series(dtype=float)

        news_data = response.json()
        if 'articles' not in news_data or not news_data['articles']:
            return pd.Series(dtype=float)

        analyzer = SentimentIntensityAnalyzer()
        sentiments = []
        for article in news_data['articles']:
            if 'publishedAt' in article and 'title' in article:
                date = article['publishedAt'][:10]
                sentiment_score = analyzer.polarity_scores(article['title'])['compound']
                sentiments.append({'date': date, 'sentiment': sentiment_score})

        if sentiments:
            sentiment_df = pd.DataFrame(sentiments)
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            return sentiment_df.groupby('date')['sentiment'].mean()
        return pd.Series(dtype=float)
    except Exception as e:
        print(f"Error fetching sentiment data for {ticker}: {e}")
        return pd.Series(dtype=float)

# Predict Future Prices
def predict_future_prices(model, seed_data, days_to_predict, feature_count, scaler):
    predictions = []
    input_sequence = seed_data[-30:]  

    for _ in range(days_to_predict):
        input_reshaped = np.reshape(input_sequence, (1, 30, feature_count))
        predicted_scaled = model.predict(input_reshaped)

        dummy_features = np.zeros((1, feature_count - 1))
        predicted_full = np.hstack((predicted_scaled, dummy_features))
        predicted_rescaled = scaler.inverse_transform(predicted_full)[0, 0]

        
        predictions.append(predicted_rescaled)

        
        last_row = input_sequence[-1]
        new_features = np.zeros(feature_count)
        new_features[0] = predicted_scaled[0, 0]  

      
        new_features[1] = np.mean(input_sequence[:, 1])  
        new_features[2] = np.mean(input_sequence[:, 2])  
        new_features[3:7] = np.mean(input_sequence[:, 3:7], axis=0)  
        new_features[7:9] = np.mean(input_sequence[:, 7:9], axis=0)  

        
        input_sequence = np.vstack((input_sequence[1:], new_features))

    return predictions


# Prepare LSTM Data
def prepare_lstm_data(stock_df, sentiment_series):
    stock_df['sentiment'] = sentiment_series.reindex(stock_df.index, fill_value=0)
    stock_df['sentiment_rolling'] = stock_df['sentiment'].rolling(window=3).mean().fillna(0)
    stock_df['MA_10'] = stock_df['Close'].rolling(window=10).mean()
    stock_df['MA_20'] = stock_df['Close'].rolling(window=20).mean()
    features = stock_df[['Close', 'sentiment', 'sentiment_rolling', 'Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_20']].fillna(0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    x, y = [], []
    for i in range(15, len(scaled_features)):
        x.append(scaled_features[i-30:i])
        y.append(scaled_features[i, 0])
    return np.array(x), np.array(y), scaler

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.get_json()
    ticker = request_data.get('ticker')
    days_to_predict = request_data.get('days', 30)

    try:
        days_to_predict = int(days_to_predict)
    except ValueError:
        return jsonify({'error': 'Invalid number of days provided.'}), 400

    stock_df = fetch_stock_data(ticker)
    sentiment_series = fetch_sentiment_data(ticker)

    if stock_df.empty:
        return jsonify({'error': 'No stock data found for the given ticker.'}), 400

    x_data, y_data, scaler = prepare_lstm_data(stock_df, sentiment_series)
    if len(x_data) == 0:
        return jsonify({'error': 'Insufficient data for LSTM model.'}), 400

    train_size = int(len(x_data) * 0.8)
    x_train, y_train = x_data[:train_size], y_data[:train_size]

    model = create_lstm_model((x_train.shape[1], x_train.shape[2]))
    model.fit(x_train, y_train, batch_size=16, epochs=100, verbose=1)

    feature_count = x_data.shape[2]
    future_predictions = predict_future_prices(model, x_data[-1], days_to_predict, feature_count, scaler)

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, days_to_predict + 1), future_predictions, label='Predicted Prices', color='teal')
    plt.title(f'{ticker} Future Stock Price Prediction ({days_to_predict} Days)')
    plt.xlabel('Days into the Future')
    plt.ylabel('Price')
    plt.legend()
    graph_path = f'static/{ticker}_future_predictions.png'
    plt.savefig(graph_path)
    plt.close()

    return jsonify({'graph_path': graph_path, 'predicted_prices': future_predictions}), 200

if __name__ == '__main__':
    app.run(debug=False)

