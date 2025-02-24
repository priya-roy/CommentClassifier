import pandas as pd
import joblib
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

from src.CommentClassifier.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config=config
        
    def load_dataset(self):
        """
        Load dataset from the transformed file.
        """
        file_path = os.path.join(self.config.data_path, "train.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"Dataset Loaded: {df.shape} rows, {df.columns.tolist()} columns")
            return df
        else:
            raise FileNotFoundError(f"Dataset not found at {file_path}")
        
    def preprocess_data(self, df):
        df = df.dropna(subset=["comment"])  # Drop missing comments
        df["comment"] = df["comment"].astype(str)  # Convert all to strings
        
        X = df[self.config.text_column]
        y = df[self.config.label_column]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        print("✅ Data Preprocessing Complete")
        return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
    
    def train_model(self, X_train_tfidf, y_train):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_tfidf, y_train)
        return model
    
    def save_model(self, model, vectorizer):
        """
        Saves the trained model and the TF-IDF vectorizer.
        """
         ## Save model
        model_path = os.path.join(self.config.root_dir, "model.pkl")
        ## Save tokenizer
        vectorizer_path = os.path.join(self.config.root_dir, "tfidf_vectorizer.pkl")

        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)

        print(f"✅ Model saved at: {model_path}")
        print(f"✅ TF-IDF Vectorizer saved at: {vectorizer_path}")