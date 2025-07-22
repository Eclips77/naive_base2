import pandas as pd
from services.evaluator.model.naive_bayes_classifier import NaiveBayesClassifier

class ModelEvaluator:
    """
    Evaluates a model on a test dataset and returns accuracy.
    """
    @staticmethod
    def evaluate(model: NaiveBayesClassifier, test_df: pd.DataFrame, target_column: str) -> float:
        """
        Evaluate the model on the test dataset.

        Args:
            model (NaiveBayesClassifier): Trained model object.
            test_df (pd.DataFrame): Test data.
            target_column (str): Name of the target column.
        Returns:
            float: Accuracy score.
        """
        y_true = test_df[target_column].tolist()
        y_pred = [model.classify(row.drop(target_column).to_dict()) for _, row in test_df.iterrows()]
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        return correct / len(y_true) if y_true else 0.0 