from flask import Flask, jsonify
import pickle
import random
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


scaler = pickle.load(open('scaler.pkl', 'rb'))
pca = pickle.load(open('pca.pkl', 'rb'))
model = pickle.load(open('rf_model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    prediction = random.choice([0, 1, 2])
    return jsonify({'prediction': int(prediction)})


if __name__ == '__main__':
    app.run(debug=True)
