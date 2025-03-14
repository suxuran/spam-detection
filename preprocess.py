import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

data = pd.read_csv('spam_dataset.csv')
data['email'] = data['email'].apply(preprocess_text)
data.to_csv('preprocessed_spam_dataset.csv', index=False)
print("Preprocessed dataset saved as preprocessed_spam_dataset.csv")