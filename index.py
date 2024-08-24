import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load the true and fake datasets
true_df = pd.read_csv('true.csv')  # Replace with the path to your true news dataset
fake_df = pd.read_csv('fake.csv')  # Replace with the path to your fake news dataset
print("Datasets loaded successfully.")

# Add a 'label' column to each dataframe
true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

# Combine the datasets
df = pd.concat([true_df, fake_df], ignore_index=True)
print("Datasets combined.")

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)
print("Dataset shuffled.")

# Display the shape of the combined dataset and first few records
print(df.shape)
print(df.head())

# Get the labels
labels = df['label']
print("Labels extracted.")

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
print("Dataset split into training and testing sets.")

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training set, transform the test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)
print("TF-IDF vectorization completed.")

# Initialize and train the PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)
print("Model training completed.")

# Predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')

# Display the confusion matrix
confusion = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print("Confusion Matrix:")
print(confusion)

# Save the model and the vectorizer (Optional)
with open('model.pkl', 'wb') as model_file:
    pickle.dump(pac, model_file)
    
with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(tfidf_vectorizer, vec_file)
print("Model and vectorizer saved.")

# Load and use the model (Optional)
def predict_fake_news(text):
    with open('model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
        
    with open('vectorizer.pkl', 'rb') as vec_file:
        loaded_vectorizer = pickle.load(vec_file)
        
    vectorized_text = loaded_vectorizer.transform([text])
    prediction = loaded_model.predict(vectorized_text)
    return prediction[0]

# Example usage of the model (Optional)
example_text = "Your article text here"
prediction = predict_fake_news(example_text)
print(f"Prediction for the example text: {prediction}")
