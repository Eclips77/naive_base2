import pandas as pd
from collections import defaultdict

class NaiveBayesClassifier:
    """
    A simple Naive Bayes classifier for categorical data.
    """
    def __init__(self):
        self.class_priors = defaultdict(float)
        self.feature_likelihoods = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.target_column = None
        self.fitted = False

    def fit(self, df: pd.DataFrame, target_column: str):
        """
        Fit the Naive Bayes classifier to the data.

        Args:
            df (pd.DataFrame): Training data.
            target_column (str): Name of the target column.
        """
        self.target_column = target_column
        class_counts = df[target_column].value_counts().to_dict()
        total = len(df)
        for cls, count in class_counts.items():
            self.class_priors[cls] = count / total
        features = [col for col in df.columns if col != target_column]
        for feature in features:
            for cls in class_counts:
                subset = df[df[target_column] == cls]
                value_counts = subset[feature].value_counts().to_dict()
                total_cls = len(subset)
                for value, count in value_counts.items():
                    self.feature_likelihoods[feature][value][cls] = count / total_cls
        self.fitted = True

    def classify(self, record: dict):
        """
        Classify a new record.

        Args:
            record (dict): Feature values for prediction.
        Returns:
            The predicted class label.
        """
        if not self.fitted:
            raise ValueError("Model is not fitted.")
        max_prob = -1
        best_class = None
        for cls in self.class_priors:
            prob = self.class_priors[cls]
            for feature, value in record.items():
                prob *= self.feature_likelihoods[feature][value].get(cls, 1e-9)
            if prob > max_prob:
                max_prob = prob
                best_class = cls
        return best_class 