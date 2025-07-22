class ModelPredictor:
    """
    Makes predictions using a loaded model.
    """
    @staticmethod
    def predict(model, record: dict):
        """
        Make a prediction using the given model and input record.

        Args:
            model: Trained model object with a 'classify' method.
            record (dict): Input features for prediction.
        Returns:
            The prediction result.
        """
        return model.classify(record) 