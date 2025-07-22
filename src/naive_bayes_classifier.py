from typing import Dict, Any
import math


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
