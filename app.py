from flask import Flask, request, jsonify
import joblib
import mysql.connector

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to save data to MySQL
def save_to_db(email, prediction):
    try:
        connection = mysql.connector.connect(
            host='localhost',      # XAMPP MySQL host
            user='root',          # Default XAMPP MySQL username
            password='',           # Default XAMPP MySQL password (empty)
            database='spam_db'     # Database name
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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email = data['email']
    email_vec = vectorizer.transform([email])
    prediction = model.predict(email_vec)
    save_to_db(email, int(prediction[0]))
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)