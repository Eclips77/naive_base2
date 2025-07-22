import os
import pickle

class ModelLoader:
    """
    Loads a trained model from a file using pickle.
    """
    @staticmethod
    def load(model_dir: str, model_name: str):
        """
        Load a model from the specified directory and file name.

        Args:
            model_dir (str): Directory containing the model.
            model_name (str): File name of the model.
        Returns:
            The loaded model object.
        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        model_path = os.path.join(model_dir, model_name)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model '{model_name}' not found in '{model_dir}'.")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model 