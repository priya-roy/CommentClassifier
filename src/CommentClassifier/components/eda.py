import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

from src.CommentClassifier.entity import DataTransformationConfig

class EDA:     
    def __init__(self,config:DataTransformationConfig):
        self.config=config
        self.text_column = self.config.text_column
        self.label_column = self.config.label_column
        self.data_path = self.config.data_path
    
    def load_dataset(self):
        """
        Load dataset from the transformed file.
        """
        file_path = os.path.join(self.config.root_dir,'commentClassification', "train.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"Dataset Loaded: {df.shape} rows, {df.columns.tolist()} columns")
            return df
        else:
            raise FileNotFoundError(f"Dataset not found at {file_path}")

    def show_data_info(self, df):
        """
        Display basic dataset information.
        """
        print("\nDataset Information:\n")
        print(df.info())
        print("\nMissing Values:\n", df.isnull().sum())
        print("\nLabel Distribution:\n", df[self.config.label_column].value_counts())

    def plot_label_distribution(self, df):
        """
        Visualise the distribution of labels (toxic vs. healthy).
        """
        plt.figure(figsize=(6,4))
        sns.countplot(x=self.config.label_column, data=df, palette="coolwarm")
        plt.title("Label Distribution")
        plt.xlabel("Healthy (1) / Toxic (0)")
        plt.ylabel("Count")
        plt.show()

    def plot_text_length_distribution(self, df):
        """
        Plot distribution of comment lengths.
        """
        df["text_length"] = df[self.config.text_column].apply(lambda x: len(str(x).split()))
        plt.figure(figsize=(8,5))
        sns.histplot(df["text_length"], bins=50, kde=True, color="blue")
        plt.title("Distribution of Comment Lengths")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")
        plt.show()

    def word_cloud(self, df, label=1):
        """
        Generate a word cloud for toxic (0) or healthy (1) comments.
        """
        text_data = " ".join(df[df[self.config.label_column] == label][self.config.text_column].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)

        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for {'Healthy' if label == 1 else 'Toxic'} Comments")
        plt.show()

    def run_eda(self):
        """
        Perform the complete EDA process.
        """
        df = self.load_dataset()
        self.show_data_info(df)
        self.plot_label_distribution(df)
        self.plot_text_length_distribution(df)
        self.word_cloud(df, label=1)  # Word cloud for healthy comments
        self.word_cloud(df, label=0)  # Word cloud for toxic comments
