import os
from services.predictor.model.model_loader import ModelLoader
from services.predictor.model.model_predictor import ModelPredictor

class PredictorManager:
    """
    Manages model loading and prediction for the Predictor service.
    """
    def __init__(self, models_dir: str):
        """
        Initialize the PredictorManager.

        Args:
            models_dir (str): Directory containing trained model files.
        """
        self.models_dir = models_dir
        self.model_loader = ModelLoader()
        self.model_predictor = ModelPredictor()

    def list_models(self):
        """
        List all available model files in the models directory.

        Returns:
            list[str]: List of model file names.
        """
        if not os.path.isdir(self.models_dir):
            return []
        return [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]

    def predict(self, model_name: str, record: dict):
        """
        Load the specified model and make a prediction for the given record.

        Args:
            model_name (str): Name of the model file.
            record (dict): Input features for prediction.
        Returns:
            The prediction result.
        """
        model = self.model_loader.load(self.models_dir, model_name)
        return self.model_predictor.predict(model, record) 