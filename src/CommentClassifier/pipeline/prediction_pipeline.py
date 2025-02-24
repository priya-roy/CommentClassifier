from src.CommentClassifier.config.configuration import ConfigurationManager
import pickle
import numpy as np
import joblib
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        """Initialises the prediction pipeline by loading the trained model and vectorizer."""
        self.config = ConfigurationManager().get_model_evaluation_config()

        # Load trained model and vectorizer
        self.model = joblib.load(self.config.model_path)
        self.vectorizer = joblib.load(self.config.tokenizer_path)

    def preprocess_text(self, text):
        """Preprocesses the input text (handles NaN, lowercase, removes special characters, etc.)."""
        if pd.isna(text):  
            text = ""  # Handle NaN values

        text = text.lower()  # Convert to lowercase (if required)
        return text

    def predict(self, text):
        """Predicts the label for a given text input."""
        # Preprocess text
        processed_text = self.preprocess_text(text)

        # Convert text to TF-IDF representation
        text_tfidf = self.vectorizer.transform([processed_text])

        # Predict the label
        prediction = self.model.predict(text_tfidf)[0]
        prediction = int(prediction)
        return {"prediction": prediction} # Return as a dictionary