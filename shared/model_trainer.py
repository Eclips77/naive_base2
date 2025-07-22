import pandas as pd
from shared.classifier import NaiveBayesClassifier

class ModelTrainer:
    """
    Trains a Naive Bayes model on a given DataFrame and target column.
    """
    @staticmethod
    def train(df: pd.DataFrame, target_column: str) -> NaiveBayesClassifier:
        """
        Train a Naive Bayes model.

        Args:
            df (pd.DataFrame): Training data.
            target_column (str): Name of the target column.
        Returns:
            NaiveBayesClassifier: Trained model.
        """
        model = NaiveBayesClassifier()
        model.fit(df, target_column)
        return model 