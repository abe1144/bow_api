from flask import Flask, request, jsonify
import json
from BoWModel import Model

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
    


if __name__ == "__main__":
    app.run(debug=True)