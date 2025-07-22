import pandas as pd
from typing import Dict, Any
from naive_bayes_classifier import NaiveBayesPredictor

class NaiveBayesEvaluator:
    """
    Evaluate model accuracy on a labelled DataFrame.

    Parameters
    ----------
    model : dict
        Serialized model dictionary.
    """

    def __init__(self, model: Dict[str, Any]) -> None:
        self.predictor = NaiveBayesPredictor(model)

    def evaluate(self, df: pd.DataFrame, target_column: str) -> float:
        """
        Calculate classification accuracy.

        Returns
        -------
        float
            Accuracy in the range [0, 1].
        """
        y_true = df[target_column].tolist()
        X = df.drop(columns=[target_column])
        y_pred = [self.predictor.predict(row.to_dict()) for _, row in X.iterrows()]
        correct = sum(p == t for p, t in zip(y_pred, y_true))
        return correct / len(y_true)
