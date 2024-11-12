from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import pandas as pd
import pickle
import webbrowser

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and required data
with open('recommender_model.pkl', 'rb') as file:
    model = pickle.load(file)

user_factors = np.loadtxt('user_factors.csv', delimiter=',')
item_factors = np.loadtxt('item_factors.csv', delimiter=',')
name_to_index_df = pd.read_csv('product_name_to_index.csv')
product_name_to_index = dict(zip(name_to_index_df['product_name'], name_to_index_df['index']))

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    product = data.get('key_product_name')

    if not product:
        return jsonify({'error': 'Invalid input, product name is required'}), 400

    try:
        if product not in product_name_to_index:
            return jsonify({'error': 'Product not found in the model data'}), 404

        idx = product_name_to_index[product]
        prediction = np.dot(user_factors[idx], item_factors[idx])
        return jsonify({'prediction': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    
    app.run(debug=True)
