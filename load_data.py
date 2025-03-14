import os
import pandas as pd

def load_dataset(data_dir):
    emails = []
    labels = []
    # Define the folders for ham and spam
    ham_folders = ['20021010_easy_ham', '20030228_easy_ham']
    spam_folders = ['20021010_spam', '20030228_spam']
    
    # Load ham emails
    for folder in ham_folders:
        dir_path = os.path.join(data_dir, folder)
        for filename in os.listdir(dir_path):
            with open(os.path.join(dir_path, filename), 'r', encoding='latin1') as f:
                emails.append(f.read())
                labels.append(0)  # 0 for ham

    # Load spam emails
    for folder in spam_folders:
        dir_path = os.path.join(data_dir, folder)
        for filename in os.listdir(dir_path):
            with open(os.path.join(dir_path, filename), 'r', encoding='latin1') as f:
                emails.append(f.read())
                labels.append(1)  # 1 for spam

    return pd.DataFrame({'email': emails, 'label': labels})

# Load the dataset
data = load_dataset('data')
data.to_csv('spam_dataset.csv', index=False)
print("Dataset saved as spam_dataset.csv")