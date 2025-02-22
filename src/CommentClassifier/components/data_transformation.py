import os
from src.CommentClassifier.logging import logger
from datasets import load_from_disk
import os
import pandas as pd
import seaborn as sns
import yaml
import nltk
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split

nltk.download("stopwords")

from src.CommentClassifier.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self,config:DataTransformationConfig):
        self.config=config
        
    def clean_text(self, text):
        """
        Perform text cleaning: remove special characters, extra spaces, and stopwords.
        """
        STOPWORDS = set(stopwords.words(self.config.stopwords))
        text = text.lower() if self.config.lowercase else text
        text = re.sub(r"[^a-zA-Z\s]", "", text) if self.config.remove_special_characters else text
        text = " ".join(text.split()) if self.config.remove_extra_spaces else text
        text = " ".join([word for word in text.split() if word not in STOPWORDS])
        return text
    
    def transform_data(self):
        """
        Load dataset, clean text, and save transformed data.
        """
        file_path = os.path.join(self.config.data_path, "train.csv")
        df = pd.read_csv(file_path)

        df[self.config.text_column] = df[self.config.text_column].astype(str).apply(self.clean_text)

        # Split the dataset into train, test, and validation sets
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)  # 80% train, 20% for temp
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)  # Split remaining 20% into 10% test, 10% validation

        # Save the transformed data into CSV files
        train_df.to_csv(os.path.join(self.config.root_dir,'commentClassification', "train.csv"), index=False)
        val_df.to_csv(os.path.join(self.config.root_dir,'commentClassification', "val.csv"), index=False)
        test_df.to_csv(os.path.join(self.config.root_dir,'commentClassification', "test.csv"), index=False)

        print("Transformed data saved successfully as train.csv, test.csv, and val.csv!")
        
        