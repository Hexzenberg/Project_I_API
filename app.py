import joblib
from flask import Flask, request, jsonify
import requests
from flask_cors import cross_origin
import os
import gdown
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)

# Function to download the models from Google Drive
def download_models():
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Google Drive file IDs for the models
    rnn_model_id = '1R5jOOgxOeKGrEGRrJcHY-0RpqqmDUijP'
    tokenizer_id = '1qbmlYwvm691kzmxlrjJIwHkquswATqWm'

    # Construct Google Drive download URLs
    rnn_model_url = f'https://drive.google.com/uc?id={rnn_model_id}'
    tokenizer_url = f'https://drive.google.com/uc?id={tokenizer_id}'

    # Download the files using gdown
    gdown.download(rnn_model_url, f'{model_dir}/rnn_model.h5', quiet=False)
    gdown.download(tokenizer_url, f'{model_dir}/tokenizer.pkl', quiet=False)

# Download the models before loading
download_models()

# Load the pre-trained RNN sentiment analysis model and Tokenizer
rnn_model = load_model('models/rnn_model.h5')  # RNN Model (loaded with Keras)
tokenizer = joblib.load('models/tokenizer.pkl')  # Tokenizer (loaded with joblib)

# Store API Key securely
API_KEY = os.getenv('NEWS_API_KEY')  # Store API Key in environment variables

# Define the news fetching route with sentiment analysis
@app.route('/get_news_with_sentiment', methods=['POST'])
@cross_origin()  # Enable CORS for this route
def get_news_with_sentiment():
    # Extract the query parameter from the POST request
    user_query = request.json.get('query')

    # Check if the user provided a query
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    # Construct the request to News API
    url = f"https://newsapi.org/v2/everything?q={user_query}&apiKey={API_KEY}"
    response = requests.get(url)

    # Handle error if the news API request fails
    if response.status_code != 200:
        return jsonify({'error': 'Unable to fetch news from the News API'}), 500

    # Get the news articles from the response
    news_data = response.json()
    articles = news_data.get('articles', [])

    if not articles:
        return jsonify({'message': 'No articles found for the query'}), 404

    # Perform sentiment analysis on each article
    analyzed_articles = []
    for article in articles:
        title = article.get('title')
        content = article.get('content', '')  # Use content if available, otherwise title

        # Choose either the content or title for sentiment input
        sentiment_input = content if content else title

        # Tokenize and pad the input for RNN model
        max_sequence_length = 100  # Use the same max length used during training
        input_sequence = tokenizer.texts_to_sequences([sentiment_input])
        padded_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length)

        # Predict the sentiment using the RNN model
        sentiment_prediction = rnn_model.predict(padded_sequence)[0][0]
        sentiment_label = 'positive' if sentiment_prediction >= 0.5 else 'negative'

        # Append the sentiment label to the article
        article['sentiment'] = sentiment_label
        analyzed_articles.append(article)

    # Return the articles with sentiment in the response
    return jsonify(analyzed_articles), 200

if __name__ == '__main__':
    app.run(debug=True)
