import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load data
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

# Preprocess data
def preprocess_data(train_df, test_df):
    # Fill missing values with empty strings
    train_df['title'].fillna('', inplace=True)
    train_df['text'].fillna('', inplace=True)
    test_df['title'].fillna('', inplace=True)
    test_df['text'].fillna('', inplace=True)
    
    # Combine title and text for feature extraction
    train_df['content'] = train_df['title'] + ' ' + train_df['text']
    test_df['content'] = test_df['title'] + ' ' + test_df['text']
    
    return train_df, test_df

# Feature extraction and model training
def feature_extraction_and_model_training(train_df):
    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(train_df['content'])
    y = train_df['label']

    # Split the training set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Validate the model
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    st.write(f'Validation Accuracy: {accuracy * 100:.2f}%')

    return vectorizer, model

# Prediction and submission
def predict_and_submit(test_df, vectorizer, model):
    X_test = vectorizer.transform(test_df['content'])
    predictions = model.predict(X_test)

    # Prepare the submission file
    submit_df = pd.DataFrame({
        'id': test_df['id'],
        'label': predictions
    })
    submit_df.to_csv('submit.csv', index=False)
    st.write('Predictions saved in submit.csv')

# Streamlit app
def streamlit_app():
    # Load the trained model and vectorizer
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # Create a text area for user input
    user_input = st.text_area("Enter the news article:", height=150)

    # Predict and display the result
    if st.button("Detect"):
        # Transform the user input using the vectorizer
        input_vector = vectorizer.transform([user_input])

        # Make prediction using the model
        prediction = model.predict(input_vector)[0]

        # Display the result
        if prediction == 0:
            st.write("The given artical is Fake.")
        else:
            st.write("The given artical is Real.")

# Run the Streamlit app
if __name__ == '__main__':
    st.title('Fake News Detection')
    streamlit_app()
