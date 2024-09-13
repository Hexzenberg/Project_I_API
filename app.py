# Steps to follow:
# pip install virtualenv
# virtualenv venv
# Activate the virtual environment - source venv/bin/activate(MacOS/Linux) or venv\Scripts\activate(Windows)
# Enter export NEWS_API_KEY='Your_API_Key' to set the API key before running.
# To run - python app.py

import joblib
from flask import Flask, request, jsonify
import requests
import os
import gdown
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initializing the Flask app.
app = Flask(__name__)

# Enabling CORS for the entire app.
CORS(app)

# Function to download the models from Google Drive.
def download_models():
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Google Drive file IDs for the models.
    rnn_model_id = '1R5jOOgxOeKGrEGRrJcHY-0RpqqmDUijP'
    tokenizer_id = '1qbmlYwvm691kzmxlrjJIwHkquswATqWm'

    # Constructing the Google Drive download URLs.
    rnn_model_url = f'https://drive.google.com/uc?id={rnn_model_id}'
    tokenizer_url = f'https://drive.google.com/uc?id={tokenizer_id}'

    try:
        # Downloading the files using gdown.
        gdown.download(rnn_model_url, f'{model_dir}/rnn_model.h5', quiet=False)
        gdown.download(tokenizer_url, f'{model_dir}/tokenizer.pkl', quiet=False)
    except Exception as e:
        print(f"Error downloading models: {e}")
        raise

# Downloading the models before loading.
download_models()

# Loading the pre-trained RNN sentiment analysis model and Tokenizer.
rnn_model = load_model('models/rnn_model.h5')  # RNN Model (loaded with Keras)
tokenizer = joblib.load('models/tokenizer.pkl')  # Tokenizer (loaded with joblib)

API_KEY = os.getenv('NEWS_API_KEY')  # Retrieving the API Key from environment variables.

# Defining the news fetching route with sentiment analysis.
@app.route('/get_news_with_sentiment', methods=['POST'])
def get_news_with_sentiment():
    # Extracting the query parameter from the POST request.
    user_query = request.json.get('query')

    # Checking if the user provided a query.
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    # Constructing the request to News API.
    url = f"https://newsapi.org/v2/everything?q={user_query}&apiKey={API_KEY}"
    response = requests.get(url)

    # Handling the error if the news API request fails.
    if response.status_code != 200:
        return jsonify({'error': 'Unable to fetch news from the News API'}), 500

    # Getting the news articles from the response.
    news_data = response.json()
    articles = news_data.get('articles', [])

    if not articles:
        return jsonify({'message': 'No articles found for the query'}), 404

    # Performing sentiment analysis on each article.
    analyzed_articles = []
    for article in articles:
        title = article.get('title')
        content = article.get('content', '')  # Use content if available, otherwise title.

        # Choosing either the content or title for sentiment input.
        sentiment_input = content if content else title

        # Tokenizing and pad the input for RNN model.
        max_sequence_length = 200  # Using the same max length used during training.
        input_sequence = tokenizer.texts_to_sequences([sentiment_input])
        padded_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length)

        # Predicting the sentiment using the RNN model.
        sentiment_prediction = rnn_model.predict(padded_sequence)[0][0]
        sentiment_label = 'positive' if sentiment_prediction >= 0.5 else 'negative'

        # Appending the sentiment label to the article.
        article['sentiment'] = sentiment_label
        analyzed_articles.append(article)

    # Returning the articles with sentiment in the response.
    return jsonify(analyzed_articles), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Using the dynamic port from Render.
    app.run(host='0.0.0.0', port=port, debug=True)
