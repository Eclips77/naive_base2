import pandas as pd
from shared.evaluation import Evaluator

class ModelEvaluator:
    """
    Evaluates a model on a test dataset and returns accuracy.
    """
    @staticmethod
    def evaluate(model, test_df: pd.DataFrame, target_column: str) -> float:
        """
        Evaluate the model on the test dataset.

        Args:
            model: Trained model object.
            test_df (pd.DataFrame): Test data.
            target_column (str): Name of the target column.
        Returns:
            float: Accuracy score.
        """
        evaluator = Evaluator()
        return evaluator.evaluate(model, test_df, target_column) 