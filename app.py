from flask import Flask, request, jsonify
import json
from BoWModel import Model
import requests
import os

app = Flask(__name__)

# with open('svm_model.pk', 'rb') as f:
#     svm_model = pickle.load(f)

svm_model = Model()

@app.route("/")
def home():
    return "<h1>Welcome to the BoW Model<h1/>"

@app.route('/predict', methods=['POST'])
def predict():
    event = json.loads(request.data)
    value = event['text']
    print(value)
    record_id, text, sentiment_prediction = svm_model.predict(value)
    sentiment_prediction = jsonify(id=str(record_id),text=str(value),sentiment=str(sentiment_prediction))
    return sentiment_prediction

@app.route('/update_data', methods=['POST'])
def update():
    event = json.loads(request.data)
    record_id = event['id']
    label = event['label']
    svm_model.update_label(record_id, label)

    response = jsonify(id=record_id, new_label=label)
    return response

@app.route('/add_record', methods=['POST'])
def add_record():
    event = json.loads(request.data)
    text = event['text']
    label = event['label']
    svm_model.update_df(text, label)
    response = jsonify(text= text, label=label)
    return response


@app.route('/get_news', methods=["POST"])
def get_news():
    event = json.loads(request.data)
    query = event['company_name']
    news_params = {"q":query,"from":"2022-03-12","to":"2022-03-22", "sortBy":"popularity","apiKey":'c9df76e6b71341ad8e61647d9876254c'}
    r = requests.get("https://newsapi.org/v2/everything", params=news_params)
    news = r.json()
    news_lst = news['articles']
    return jsonify(articles=news_lst)


if __name__ == "__main__":
    app.run(debug=True)