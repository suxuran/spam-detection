from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import mysql.connector
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Enable CORS for specific origins
CORS(app, resources={
    r"/predict": {
        "origins": ["https://suxuran.github.io"]
    }
})

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to save data to MySQL
def save_to_db(email, prediction):
    try:
        connection = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST'),
            user=os.getenv('MYSQL_USER'),
            password=os.getenv('MYSQL_PASSWORD'),
            database=os.getenv('MYSQL_DATABASE')
        )
        cursor = connection.cursor()
        query = "INSERT INTO emails (email, label) VALUES (%s, %s)"
        cursor.execute(query, (email, prediction))
        connection.commit()
        cursor.close()
        connection.close()
        print("Data saved to MySQL!")
    except Exception as e:
        print(f"Error: {e}")

# Root endpoint
@app.route('/')
def home():
    return "Spam Email Detection API is running!"

# Predict endpoint
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Handle preflight requests
        response = jsonify({'status': 'success'})
        response.headers.add('Access-Control-Allow-Origin', 'https://suxuran.github.io')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    data = request.json
    email = data['email']
    email_vec = vectorizer.transform([email])
    prediction = model.predict(email_vec)
    save_to_db(email, int(prediction[0]))
    response = jsonify({'prediction': int(prediction[0])})
    response.headers.add('Access-Control-Allow-Origin', 'https://suxuran.github.io')
    return response

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)