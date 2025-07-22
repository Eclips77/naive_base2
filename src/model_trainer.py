import json
import pandas as pd
from typing import Dict, Any, List


class NaiveBayesTrainer:
    """
    Train a categorical Naive Bayes model and serialize it to JSON.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset that includes both features and the target column.
    target_column : str
        Name of the label column.

    Attributes
    ----------
    model_ : dict
        Nested dictionary holding priors and conditional probabilities.
    classes_ : List[Any]
        List of unique class labels discovered during fitting.
    """

    def __init__(self) -> None:
        self.model_: Dict[str, Any] = {}
        self.classes_: List[Any] = []

    def fit(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Fit the model and return the in‑memory representation.

        Returns
        -------
        dict
            The trained model dictionary.
        """
        y = df[target_column]
        X = df.drop(columns=[target_column])

        self.classes_ = y.unique().tolist()
        priors = y.value_counts(normalize=True).to_dict()
        model: Dict[str, Any] = {"priors": priors, "likelihoods": {}, "classes": self.classes_}

        for col in X.columns:
            likelihoods_col: Dict[Any, Dict[Any, float]] = {}
            crosstab = (pd.crosstab(X[col], y) + 1).div(len(self.classes_))  # Laplace
            totals = crosstab.sum()
            likelihoods = crosstab.div(totals)
            for feature_val, row in likelihoods.iterrows():
                likelihoods_col[feature_val] = row.to_dict()
            model["likelihoods"][col] = likelihoods_col

        self.model_ = model
        return model

    @staticmethod
    def save_model(model: Dict[str, Any], path: str) -> None:
        """
        Persist the trained model to disk as JSON.

        Parameters
        ----------
        model : dict
            The trained model dictionary.
        path : str
            Destination file path (e.g. "models/model.json").
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_model(path: str) -> Dict[str, Any]:
        """
        Load a model dictionary from a JSON file.

        Returns
        -------
        dict
            The deserialized model.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


