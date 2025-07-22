from typing import Dict, Any
import math
import pandas as pd


class NaiveBayesPredictor:
    """
    Predict class labels for categorical samples using a serialized Naive Bayes model.

    Parameters
    ----------
    model : dict
        Dictionary produced by `NaiveBayesTrainer.fit` or `load_model` containing:
        - "priors": prior probabilities for each class
        - "likelihoods": nested mapping feature → value → class → P(value|class)
        - "classes": list of class labels
    """

    _EPS = 1e-9  # minimal probability to avoid log(0)

    def __init__(self, model: Dict[str, Any]) -> None:
        self.model = model
        self.classes = model["classes"]
        self.priors = model["priors"]
        self.likelihoods = model["likelihoods"]

    def _safe_log(self, p: float) -> float:
        """Return log‑probability, clamped to avoid −inf."""
        return math.log(p if p > 0 else self._EPS)

    def predict(self, sample: Dict[str, Any]) -> Any:
        """
        Predict the most probable class for a single sample.

        Parameters
        ----------
        sample : dict
            Mapping of feature name → categorical value.

        Returns
        -------
        Any
            Predicted class label.
        """
        # initialise with log‑priors
        log_probs = {
            c: self._safe_log(self.priors.get(str(c), self._EPS)) for c in self.classes
        }

        # accumulate log‑likelihoods per feature
        for feature, value in sample.items():
            value_likelihoods = self.likelihoods.get(feature, {})
            for c in self.classes:
                prob = value_likelihoods.get(value, {}).get(str(c), self._EPS)
                log_probs[c] += self._safe_log(prob)

        # return class with highest posterior
        return max(log_probs, key=log_probs.get)


class NaiveBayesClassifier:
    """High level wrapper combining training and prediction."""

    def __init__(self) -> None:
        self.model: Dict[str, Any] = {}

    def fit(self, df: "pd.DataFrame", target_column: str) -> Dict[str, Any]:
        """Train a Naive Bayes model on ``df`` using ``target_column``."""
        from .model_trainer import NaiveBayesTrainer

        trainer = NaiveBayesTrainer()
        self.model = trainer.fit(df, target_column)
        return self.model

    def classify(self, record: Dict[str, Any]) -> Any:
        """Predict a single ``record`` using the trained model."""
        if not self.model:
            raise ValueError("Model has not been trained yet.")

        predictor = NaiveBayesPredictor(self.model)
        return predictor.predict(record)

