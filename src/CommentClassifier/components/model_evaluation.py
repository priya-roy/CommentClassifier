import joblib
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import os

from src.CommentClassifier.entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.model = self.load_model()
        self.vectorizer = self.load_vectorizer()
        
    def load_model(self):
        """ Load the trained model """
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return joblib.load(model_path)

    def load_vectorizer(self):
        """ Load the saved TF-IDF vectorizer """
        vectorizer_path = Path(self.config.tokenizer_path)
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {tokenizer_path}")
        return joblib.load(vectorizer_path)

    def load_data(self):
        """ Load validation and test datasets """
        data_path = Path(self.config.data_path, "train.csv")

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)
        # Fill NaN values with an empty string
        df[self.config.text_column] = df[self.config.text_column].fillna("")
        X = df[self.config.text_column]
        y = df[self.config.label_column]

        # Splitting into validation and test sets
        split_idx = int(len(df) * (1 - self.config.validation_split))
        X_validation, y_validation = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]

        return X_validation, y_validation, X_test, y_test

    def evaluate_model(self, model, vectorizer, X, y):
        """ Transform text using TF-IDF and evaluate the model """
        X_tfidf = vectorizer.transform(X)  # Apply TF-IDF transformation
        predictions = model.predict(X_tfidf)
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions)
        return accuracy, report
    
    def save_metrics(self, metrics):
        """Save metrics as a CSV file."""
        metrics_path = self.config.metric_file_name
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved at {metrics_path}")

    def run_evaluation(self):
        """ Run model evaluation """
        print("Loading model and vectorizer...")
        model = self.load_model()
        vectorizer = self.load_vectorizer()

        print("Loading data...")
        X_validation, y_validation, X_test, y_test = self.load_data()

        print("Evaluating on Validation Set...")
        val_acc, val_report = self.evaluate_model(model, vectorizer, X_validation, y_validation)
        print(f"Validation Accuracy: {val_acc}")
        print(f"Validation Classification Report:\n{val_report}")

        print("Evaluating on Test Set...")
        test_acc, test_report = self.evaluate_model(model, vectorizer, X_test, y_test)
        print(f"Test Accuracy: {test_acc}")
        print(f"Test Classification Report:\n{test_report}")

        # Save metrics
        metrics = {
            "Validation Accuracy": val_acc,
            "Validation Report": val_report,
            "Test Accuracy": test_acc,
            "Test Report": test_report
        }
        self.save_metrics(metrics)