from flask import Flask, request, jsonify
import joblib
import mysql.connector
import os
from dotenv import load_dotenv  # For loading environment variables

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to save data to MySQL
def save_to_db(email, prediction):
    try:
        # Use environment variables for database connection
        connection = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST'),      # MySQL host
            user=os.getenv('MYSQL_USER'),      # MySQL username
            password=os.getenv('MYSQL_PASSWORD'),  # MySQL password
            database=os.getenv('MYSQL_DATABASE')   # Database name
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
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email = data['email']
    email_vec = vectorizer.transform([email])
    prediction = model.predict(email_vec)
    save_to_db(email, int(prediction[0]))
    return jsonify({'prediction': int(prediction[0])})

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT if provided, otherwise default to 5000
    app.run(host='0.0.0.0', port=port)